"""
GNN 模型定义 — GCN / GAT / HGT。

支持 PyG (PyTorch Geometric) 和纯 NumPy 降级实现。
"""

import logging
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)

# 检测 PyG 可用性
_PYG_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _PYG_AVAILABLE = True
except ImportError:
    _logger.warning("PyTorch not available — GNN models will use NumPy fallback")


from tushenjing.gnn_numpy import NumPyGCN, NumPyGAT

# ── PyG 模型（当 PyTorch 可用时）───────────────────────

if _PYG_AVAILABLE:

    class PyGGCN(nn.Module):
        """PyG 实现的 GCN 模型。"""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
            super().__init__()
            self.conv1 = None  # 延迟导入 PyG
            self.conv2 = None
            self.dropout = dropout
            self._in_dim = in_dim
            self._hidden_dim = hidden_dim
            self._out_dim = out_dim
            self._init_layers()

        def _init_layers(self):
            try:
                from torch_geometric.nn import GCNConv
                self.conv1 = GCNConv(self._in_dim, self._hidden_dim)
                self.conv2 = GCNConv(self._hidden_dim, self._out_dim)
            except ImportError:
                _logger.warning("torch_geometric not available, GCN will not work")

        def forward(self, x, edge_index):
            if self.conv1 is None:
                return torch.zeros(x.size(0), self._out_dim)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x

        def get_weights_snapshot(self) -> dict:
            """获取当前权重快照（用于早停恢复）。"""
            return {k: v.clone() for k, v in self.state_dict().items()}

        def load_weights_snapshot(self, snapshot: dict) -> None:
            """加载权重快照。"""
            self.load_state_dict(snapshot)

    class PyGGAT(nn.Module):
        """PyG 实现的 GAT 模型。"""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.5):
            super().__init__()
            self._in_dim = in_dim
            self._hidden_dim = hidden_dim
            self._out_dim = out_dim
            self._heads = heads
            self.dropout = dropout
            self.conv1 = None
            self.conv2 = None
            self._init_layers()

        def _init_layers(self):
            try:
                from torch_geometric.nn import GATConv
                self.conv1 = GATConv(self._in_dim, self._hidden_dim, heads=self._heads, dropout=self.dropout)
                self.conv2 = GATConv(self._hidden_dim * self._heads, self._out_dim, heads=1, concat=False, dropout=self.dropout)
            except ImportError:
                _logger.warning("torch_geometric not available, GAT will not work")

        def forward(self, x, edge_index):
            if self.conv1 is None:
                return torch.zeros(x.size(0), self._out_dim)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x

        def get_weights_snapshot(self) -> dict:
            """获取当前权重快照（用于早停恢复）。"""
            return {k: v.clone() for k, v in self.state_dict().items()}

        def load_weights_snapshot(self, snapshot: dict) -> None:
            """加载权重快照。"""
            self.load_state_dict(snapshot)


# ── 工厂函数 ──────────────────────────────────────────

def create_model(
    model_type: str,       # "gcn" | "gat" | "hgt"
    in_dim: int,
    hidden_dim: int = 64,
    out_dim: int = 1,
    **kwargs,
):
    """创建 GNN 模型。

    优先使用 PyG 实现（GPU 加速），不可用时降级为 NumPy。

    Args:
        model_type: 模型类型
        in_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        out_dim: 输出维度（通常 1 = 增强因子得分）

    Returns:
        模型对象
    """
    if _PYG_AVAILABLE and model_type == "gcn":
        return PyGGCN(in_dim, hidden_dim, out_dim, kwargs.get("dropout", 0.5))
    elif _PYG_AVAILABLE and model_type == "gat":
        return PyGGAT(in_dim, hidden_dim, out_dim, kwargs.get("heads", 4), kwargs.get("dropout", 0.5))

    # NumPy 降级 (完整参数对齐 PyG 版本)
    dropout = kwargs.get("dropout", 0.5)
    if model_type == "gcn":
        return NumPyGCN(in_dim, hidden_dim, out_dim, dropout=dropout)
    elif model_type == "gat":
        return NumPyGAT(in_dim, hidden_dim, out_dim, heads=kwargs.get("heads", 1), dropout=dropout)
    elif model_type == "hgt":
        raise ValueError("HGT not supported in NumPy fallback — use GCN or GAT")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _dense_to_edge_index(adj):
    """Convert dense [N,N] adjacency matrix to PyG edge_index [2, E]."""
    import torch
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj).float()
    rows, cols = torch.nonzero(adj, as_tuple=True)
    return torch.stack([rows, cols], dim=0).long()


def _ensure_tensor(x):
    """Convert numpy array to torch tensor if needed."""
    import torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


class GCNModel:
    """GCN 模型包装器。自动处理 dense adj -> edge_index 转换。"""

    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        self._model = create_model("gcn", in_dim, hidden_dim, out_dim)
        self._is_pyg = _PYG_AVAILABLE and isinstance(self._model, PyGGCN) if _PYG_AVAILABLE else False

    def __call__(self, x, adj_norm):
        if self._is_pyg:
            x_t = _ensure_tensor(x)
            ei = _dense_to_edge_index(adj_norm)
            return self._model.forward(x_t, ei)
        return self._model.forward(x, adj_norm)

    @property
    def model(self):
        return self._model


class GATModel:
    """GAT 模型包装器 — 同时支持 PyG 和 NumPy 降级。"""

    def __init__(self, in_dim, hidden_dim=64, out_dim=1, heads=4, dropout=0.5):
        self._model = create_model("gat", in_dim, hidden_dim, out_dim, heads=heads, dropout=dropout)
        self._is_pyg = _PYG_AVAILABLE and isinstance(self._model, PyGGAT) if _PYG_AVAILABLE else False

    def __call__(self, x, adj):
        if self._is_pyg:
            x_t = _ensure_tensor(x)
            ei = _dense_to_edge_index(adj)
            return self._model.forward(x_t, ei)
        return self._model.forward(x, adj)

    @property
    def model(self):
        return self._model
