"""
GNN 模型训练器。

训练循环、验证、早停、学习率调度。
"""

import logging

import numpy as np

from tushenjing.metrics import gnn_early_stopped, gnn_train_epochs_total, gnn_train_loss, gnn_val_loss

_logger = logging.getLogger(__name__)


class GraphTrainer:
    """GNN 模型训练器。

    支持两种损失模式:
      - MSE (默认): fit()
      - 自定义损失: fit_ranking() — 用于股票排序场景
    """

    def __init__(
        self,
        model,
        learning_rate: float = 0.01,
        epochs: int = 200,
        patience: int = 20,
        verbose: bool = False,
    ):
        self.model = model
        self.lr = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    # ── 训练 (MSE) ──────────────────────────────────────

    def fit(
        self,
        x: np.ndarray,            # [N, F] 特征
        adj_norm: np.ndarray,     # [N, N] 归一化邻接矩阵
        y: np.ndarray,            # [N] 或 [N, 1] 标签
        train_mask: np.ndarray,   # [N] bool
        val_mask: np.ndarray,     # [N] bool
        loss_fn=None,             # 可选自定义损失函数
    ) -> dict:
        """训练 GNN 模型。

        Args:
            loss_fn: 可选自定义损失函数 (preds, labels) -> tensor。
                     若为 None 则使用 MSE。

        Returns:
            {train_losses, val_losses, best_epoch, early_stopped}
        """
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        y = y.reshape(-1, 1) if y.ndim == 1 else y

        for epoch in range(1, self.epochs + 1):
            y_pred = self.model(x, adj_norm)

            if loss_fn is not None:
                train_loss = float(loss_fn(y_pred[train_mask], y[train_mask]))
                val_loss = float(loss_fn(y_pred[val_mask], y[val_mask]))
            else:
                train_loss = self._mse(y_pred[train_mask], y[train_mask])
                val_loss = self._mse(y_pred[val_mask], y[val_mask])

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights = self._get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            gnn_train_loss.set(train_loss)
            gnn_val_loss.set(val_loss)
            gnn_train_epochs_total.inc()

            if patience_counter >= self.patience:
                if self.verbose:
                    _logger.info("Early stopping at epoch %d (best val_loss=%.6f)", epoch, best_val_loss)
                break

        if best_weights is not None:
            self._load_weights(best_weights)

        gnn_early_stopped.set(1 if patience_counter >= self.patience else 0)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": len(self.val_losses) - patience_counter,
            "early_stopped": patience_counter >= self.patience,
        }

    # ── 逐 batch 训练（多快照时序数据）───────────────────

    def fit_batches(
        self,
        features_list: list[np.ndarray],
        labels_list: list[np.ndarray],
        edge_index,  # torch Tensor [2, E]
        n_train: int,
        optimizer,   # torch Optimizer
        loss_fn,     # (preds, labels) -> tensor
        device,
    ) -> dict:
        """对多快照时序数据执行逐 batch 训练。

        每个 epoch 遍历 n_train 个快照，每 batch 执行一次前向+反向传播。
        用于 train_gnn_model.py 的迁移目标。

        Args:
            features_list: 每个快照的特征 [N, F]
            labels_list: 每个快照的标签 [N]
            edge_index: PyG 边索引 [2, E]
            n_train: 训练快照数 (前 n_train 个用于训练)
            optimizer: torch 优化器
            loss_fn: 损失函数 (preds, labels) -> scalar tensor
            device: torch device

        Returns:
            {epochs_completed, train_losses}
        """
        import torch

        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tl, nb = 0.0, 0
            for t in range(n_train):
                x = torch.from_numpy(features_list[t]).float().to(device)
                y = torch.from_numpy(labels_list[t]).float().view(-1).to(device)
                vm = ~torch.isnan(y)
                if vm.sum() < 50:
                    continue
                preds = self.model(x, edge_index).view(-1)
                loss = loss_fn(preds[vm], y[vm])
                if loss.item() == 0:
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tl += loss.item()
                nb += 1

            avg_loss = tl / nb if nb > 0 else 0
            train_losses.append(avg_loss)
            self.train_losses.append(avg_loss)

            if self.verbose and epoch % 20 == 0 and nb > 0:
                _logger.info("Epoch %3d: rank_loss=%.6f", epoch, avg_loss)

        gnn_train_epochs_total.inc(self.epochs)
        return {"epochs_completed": self.epochs, "train_losses": train_losses}

    # ── 内部 ────────────────────────────────────────────

    @staticmethod
    def _mse(y_pred, y_true) -> float:
        """均方误差。兼容 numpy 和 torch tensor。"""
        try:
            import torch
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().numpy()
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().numpy()
        except ImportError:
            pass
        return float(np.mean((y_pred - y_true) ** 2))

    def _get_weights(self) -> dict | None:
        """获取当前模型权重快照。"""
        if hasattr(self.model, 'get_weights_snapshot'):
            return self.model.get_weights_snapshot()
        return None

    def _load_weights(self, snapshot: dict) -> None:
        """加载权重快照。"""
        if hasattr(self.model, 'load_weights_snapshot'):
            self.model.load_weights_snapshot(snapshot)
