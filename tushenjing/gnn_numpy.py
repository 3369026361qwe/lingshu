"""
纯 NumPy GNN 实现 — GCN + GAT（多头注意力 + 向量化）。

当 PyTorch/PyG 不可用时的完整降级方案。
与 PyG 版本关键对齐:
  - Xavier 初始化 (与 PyG 默认一致)
  - 多头注意力 (GAT, heads=1~8)
  - 权重快照 + 从 PyG checkpoint 加载
  - 数值稳定 (softmax 减 max, LeakyReLU)

Usage:
    gcn = NumPyGCN(in_dim=20, hidden_dim=64, out_dim=1)
    scores = gcn.forward(features, adj_norm)  # [N, F] @ [N, N] → [N, out_dim]
"""

import numpy as np


# ── 初始化工具 ─────────────────────────────────────────

def _xavier_init(fan_in: int, fan_out: int, shape: tuple, rng: np.random.RandomState) -> np.ndarray:
    """Xavier/Glorot 均匀初始化，与 PyG 默认一致。"""
    scale = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-scale, scale, shape).astype(np.float32)


def _xavier_init_stack(weights: list[tuple[int, int, tuple]], seed: int = 42) -> list[np.ndarray]:
    """批量 Xavier 初始化。"""
    rng = np.random.RandomState(seed)
    return [_xavier_init(fi, fo, sh, rng) for fi, fo, sh in weights]


# ── NumPy GCN ──────────────────────────────────────────

class NumPyGCN:
    """纯 NumPy GCN — Xavier 初始化 + 两层图卷积。

    与 PyGGCN 对齐：ReLU 激活，dropout 在 NumPy 下退化为恒等。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5, seed: int = 42):
        self.dropout = dropout
        self._seed = seed
        W1, W2 = _xavier_init_stack([
            (in_dim, hidden_dim, (in_dim, hidden_dim)),
            (hidden_dim, out_dim, (hidden_dim, out_dim)),
        ], seed)
        self.W1 = W1
        self.W2 = W2
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
        """前向传播: H = ReLU(Â @ X @ W1) @ Â @ W2。"""
        h = adj_norm @ x @ self.W1 + self.b1
        h = np.maximum(0, h)  # ReLU
        if self.dropout > 0:
            # NumPy 降级: inference 模式下不做 dropout
            # 训练模式下随机失活（可选启用）
            pass
        h = adj_norm @ h @ self.W2 + self.b2
        return h

    def get_weights_snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "W2": self.W2.copy(),
                "b1": self.b1.copy(), "b2": self.b2.copy()}

    def load_weights_snapshot(self, snapshot: dict) -> None:
        self.W1 = snapshot["W1"]; self.W2 = snapshot["W2"]
        self.b1 = snapshot.get("b1", self.b1); self.b2 = snapshot.get("b2", self.b2)


# ── NumPy GAT (多头注意力) ─────────────────────────────

class NumPyGAT:
    """纯 NumPy GAT — 多头注意力 + LeakyReLU + softmax。

    与 PyGGAT 对齐:
      e_ij = LeakyReLU( a_left^T @ Wh_i + a_right^T @ Wh_j )
      α_ij = softmax_j(e_ij)
      h'_i = ||_{k=1..heads} Σ_j α_ij^k @ W^k h_j

    heads=1 → 单头 (与旧版兼容)
    heads=4 → 4 头拼接后线性投影到 out_dim
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 heads: int = 1, dropout: float = 0.5, seed: int = 42):
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.leaky_relu_slope = 0.2
        rng = np.random.RandomState(seed)

        # 每头独立的 W, a_left, a_right
        self.W_heads = []
        self.a_left_heads = []
        self.a_right_heads = []
        for _ in range(heads):
            self.W_heads.append(_xavier_init(in_dim, hidden_dim, (in_dim, hidden_dim), rng))
            self.a_left_heads.append(_xavier_init(hidden_dim, 1, (hidden_dim, 1), rng))
            self.a_right_heads.append(_xavier_init(hidden_dim, 1, (hidden_dim, 1), rng))

        # 输出投影
        self.W_out = _xavier_init(hidden_dim * heads, out_dim, (hidden_dim * heads, out_dim), rng)

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """多头注意力前向传播。

        Args:
            x: 节点特征 [N, in_dim]
            adj: 邻接矩阵 [N, N] (>0 表示边)

        Returns:
            h_out: [N, out_dim]
        """
        N, _ = x.shape
        head_outputs = []

        for k in range(self.heads):
            W = self.W_heads[k]
            a_left = self.a_left_heads[k]
            a_right = self.a_right_heads[k]
            h = x @ W  # [N, hidden_dim]
            h_head = np.zeros((N, self.hidden_dim), dtype=np.float32)

            # 向量化计算每对 (i, j) 的注意力
            for i in range(N):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) == 0:
                    h_head[i] = h[i]
                else:
                    h_i = h[i]
                    h_neighbors = h[neighbors]  # [deg(i), hidden_dim]
                    # 批量计算 e_ij
                    e_left = h_i @ a_left  # scalar
                    e_right = h_neighbors @ a_right  # [deg(i), 1]
                    e_ij = e_left + e_right.squeeze(1)  # [deg(i)]
                    # LeakyReLU
                    e_ij = np.where(e_ij > 0, e_ij, self.leaky_relu_slope * e_ij)
                    # Softmax
                    e_ij = np.exp(e_ij - e_ij.max())
                    alpha = e_ij / e_ij.sum()
                    # 加权聚合
                    h_head[i] = alpha @ h_neighbors

            head_outputs.append(h_head)

        # 多头拼接
        h_cat = np.concatenate(head_outputs, axis=1)  # [N, hidden_dim * heads]
        return h_cat @ self.W_out

    def get_weights_snapshot(self) -> dict:
        snap = {"W_out": self.W_out.copy()}
        for k in range(self.heads):
            snap[f"W_{k}"] = self.W_heads[k].copy()
            snap[f"a_left_{k}"] = self.a_left_heads[k].copy()
            snap[f"a_right_{k}"] = self.a_right_heads[k].copy()
        return snap

    def load_weights_snapshot(self, snapshot: dict) -> None:
        self.W_out = snapshot["W_out"]
        for k in range(self.heads):
            if f"W_{k}" in snapshot:
                self.W_heads[k] = snapshot[f"W_{k}"]
                self.a_left_heads[k] = snapshot[f"a_left_{k}"]
                self.a_right_heads[k] = snapshot[f"a_right_{k}"]


# ── PyG Checkpoint → NumPy 权重加载 ────────────────────

def load_pyg_checkpoint_to_numpy(checkpoint_path: str) -> dict:
    """从 PyG checkpoint (.pt) 提取权重并转换为 NumPy 模型。

    将 PyTorch 训练好的 GCN/GAT 权重映射到 NumPyGCN/NumPyGAT 的参数空间。
    仅支持单头 GAT → NumPyGAT(heads=1) 的转换。

    Args:
        checkpoint_path: .pt 文件路径 (如 data/gnn_model.pt)

    Returns:
        {model_type, state_dict, features, stock_codes, ...}
        或空 dict (转换失败时)
    """
    import json
    from pathlib import Path

    if not Path(checkpoint_path).exists():
        return {}

    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except (ImportError, Exception):
        return {}

    model_type = ckpt.get('model_type', 'GCN')
    in_dim = len(ckpt.get('features', []))
    hidden_dim = ckpt.get('hidden_dim', 64)
    out_dim = 1
    dropout = ckpt.get('dropout', 0.5)
    heads = ckpt.get('gat_heads', 1) if model_type == 'GAT' else 1

    state_dict = ckpt.get('model_state_dict', {})
    if not state_dict:
        return {}

    # 创建 NumPy 模型
    if model_type == 'GAT':
        model = NumPyGAT(in_dim, hidden_dim, out_dim, heads=heads, dropout=dropout)
    else:
        model = NumPyGCN(in_dim, hidden_dim, out_dim, dropout=dropout)

    # 映射权重: PyG conv1.lin_src.weight → NumPy W1, etc.
    numpy_snapshot = _map_pyg_to_numpy(state_dict, model_type, heads)
    if numpy_snapshot:
        if model_type == 'GAT':
            model.load_weights_snapshot(numpy_snapshot)
        else:
            model.load_weights_snapshot(numpy_snapshot)

    return {
        'model': model,
        'model_type': model_type,
        'stock_codes': ckpt.get('stock_codes', []),
        'edge_index': ckpt.get('edge_index', None),
        'features': ckpt.get('features', []),
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'heads': heads,
    }


def _map_pyg_to_numpy(state_dict: dict, model_type: str, heads: int) -> dict:
    """将 PyG state_dict 映射为 NumPy 模型的权重快照。

    PyG GCNConv 参数:
      conv1.lin.weight → W1  [hidden_dim, in_dim] → 转置后 [in_dim, hidden_dim]
      conv1.bias → b1      [hidden_dim]

    PyG GATConv 参数 (per head):
      conv1.lin_src.weight  → W_0  (第 0 头的 W)
      conv1.att_src         → a_left_0
      conv1.att_dst         → a_right_0
    """
    snapshot = {}
    try:
        import torch

        if model_type == 'GCN':
            # conv1.lin.weight: [hidden_dim, in_dim] → NumPy [in_dim, hidden_dim]
            if 'conv1.lin.weight' in state_dict:
                snapshot['W1'] = state_dict['conv1.lin.weight'].cpu().numpy().T.astype(np.float32)
            if 'conv1.bias' in state_dict:
                snapshot['b1'] = state_dict['conv1.bias'].cpu().numpy().astype(np.float32)
            if 'conv2.lin.weight' in state_dict:
                snapshot['W2'] = state_dict['conv2.lin.weight'].cpu().numpy().T.astype(np.float32)
            if 'conv2.bias' in state_dict:
                snapshot['b2'] = state_dict['conv2.bias'].cpu().numpy().astype(np.float32)

        elif model_type == 'GAT':
            # Per-head mapping — PyG 2.x 用 conv1.lin.weight (不是 konv1.lin_src)
            lin_key = 'conv1.lin_src.weight' if 'conv1.lin_src.weight' in state_dict else 'conv1.lin.weight'
            hd = state_dict.get(lin_key)
            if hd is not None:
                hd = hd.cpu().numpy()
                # GATConv stores [heads * hidden, in_dim]
                per_head_dim = hd.shape[0] // heads
                for k in range(heads):
                    snapshot[f'W_{k}'] = hd[k * per_head_dim:(k + 1) * per_head_dim, :].T.astype(np.float32)

            for k in range(heads):
                for src_dst, key in [('att_src', f'a_left_{k}'), ('att_dst', f'a_right_{k}')]:
                    ak = f'conv1.{src_dst}'
                    if ak in state_dict:
                        ak_data = state_dict[ak].cpu().numpy()
                        if ak_data.ndim == 1:
                            snapshot[key] = ak_data.reshape(-1, 1).astype(np.float32)
                        elif ak_data.ndim == 3:
                            # PyG >=2.5: [num_edge_types=1, heads, hidden]
                            snapshot[key] = ak_data[0, k, :].reshape(-1, 1).astype(np.float32)
                        else:
                            per_h = ak_data.shape[0] // heads
                            snapshot[key] = ak_data[k * per_h:(k + 1) * per_h].reshape(-1, 1).astype(np.float32)

            # Output projection — PyG 2.x 用 conv2.lin.weight (不是 conv2.lin_src)
            w_out_key = 'conv2.lin_src.weight' if 'conv2.lin_src.weight' in state_dict else 'conv2.lin.weight'
            if w_out_key in state_dict:
                snapshot['W_out'] = state_dict[w_out_key].cpu().numpy().T.astype(np.float32)

    except Exception:
        return {}

    return snapshot if snapshot else {}
