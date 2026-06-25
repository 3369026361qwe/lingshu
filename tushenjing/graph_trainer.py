"""
GNN 模型训练器。

训练循环、验证、早停、学习率调度。
"""

import logging
from typing import Optional

import numpy as np

from tushenjing.metrics import gnn_train_loss, gnn_val_loss, gnn_train_epochs_total, gnn_early_stopped

_logger = logging.getLogger(__name__)


class GraphTrainer:
    """GNN 模型训练器。"""

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

    # ── 训练 ────────────────────────────────────────────

    def fit(
        self,
        x: np.ndarray,            # [N, F] 特征
        adj_norm: np.ndarray,     # [N, N] 归一化邻接矩阵
        y: np.ndarray,            # [N] 或 [N, 1] 标签
        train_mask: np.ndarray,   # [N] bool
        val_mask: np.ndarray,     # [N] bool
    ) -> dict:
        """训练 GNN 模型。

        Returns:
            {train_losses, val_losses, best_epoch, early_stopped}
        """
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        y = y.reshape(-1, 1) if y.ndim == 1 else y

        for epoch in range(1, self.epochs + 1):
            # 前向传播
            y_pred = self.model(x, adj_norm)

            # 训练损失 (MSE)
            train_loss = self._mse(y_pred[train_mask], y[train_mask])
            self.train_losses.append(train_loss)

            # 验证损失
            val_loss = self._mse(y_pred[val_mask], y[val_mask])
            self.val_losses.append(val_loss)

            # 早停检查
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights = self._get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            # P0-1: 上报 Prometheus 指标
            gnn_train_loss.set(train_loss)
            gnn_val_loss.set(val_loss)
            gnn_train_epochs_total.inc()

            if patience_counter >= self.patience:
                if self.verbose:
                    _logger.info("Early stopping at epoch %d (best val_loss=%.6f)", epoch, best_val_loss)
                break

        # 恢复最佳权重
        if best_weights is not None:
            self._load_weights(best_weights)

        gnn_early_stopped.set(1 if patience_counter >= self.patience else 0)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": len(self.val_losses) - patience_counter,
            "early_stopped": patience_counter >= self.patience,
        }

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

    def _get_weights(self) -> Optional[dict]:
        """获取当前模型权重快照。"""
        if hasattr(self.model, 'get_weights_snapshot'):
            return self.model.get_weights_snapshot()
        return None

    def _load_weights(self, snapshot: dict) -> None:
        """加载权重快照。"""
        if hasattr(self.model, 'load_weights_snapshot'):
            self.model.load_weights_snapshot(snapshot)
