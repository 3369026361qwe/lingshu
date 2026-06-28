"""
图神经网络推理引擎。

对训练好的 GNN 模型执行推理，产出信息增强的因子得分。
"""

import logging
import time as _time

import numpy as np

from tushenjing.graph_utils import GraphUtils
from tushenjing.metrics import gnn_inference_duration, gnn_inference_nodes_total

_logger = logging.getLogger(__name__)


class GraphInference:
    """GNN 推理引擎。"""

    def __init__(self, model, normalize_adj: bool = True):
        self.model = model
        self.normalize_adj = normalize_adj

    # ── 推理 ────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,      # [N, F]
        adj: np.ndarray,            # [N, N]
        stock_list: list[str],
    ) -> dict[str, float]:
        """执行推理，返回每只股票的 GNN 增强得分。

        Args:
            features: 节点特征矩阵
            adj: 邻接矩阵
            stock_list: 股票代码列表

        Returns:
            {code: gnn_score}
        """
        t0 = _time.perf_counter()
        # 归一化邻接矩阵
        if self.normalize_adj:
            adj_norm = GraphUtils.normalize_adjacency(adj)
        else:
            adj_norm = adj

        # 前向传播
        scores = self.model(features, adj_norm)

        # 转为字典
        if hasattr(scores, 'detach'):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).ravel()
        result = {code: float(scores[i]) for i, code in enumerate(stock_list) if i < len(scores)}
        gnn_inference_duration.observe(_time.perf_counter() - t0)
        gnn_inference_nodes_total.inc(len(stock_list))
        return result

    def predict_batch(
        self,
        graph_data: dict,
        factor_data: dict[str, dict],
        feature_order: list[str] | None = None,
    ) -> dict[str, float]:
        """一步式推理：从图数据 + 因子数据 → GNN 增强得分。

        Args:
            graph_data: GraphBuilder.build() 的输出
            factor_data: {code: {factor: value}}
            feature_order: 特征列顺序

        Returns:
            {code: gnn_enhanced_score}
        """
        stock_list = graph_data["nodes"]
        features, _ = GraphUtils.build_feature_matrix(stock_list, factor_data, feature_order)
        features = GraphUtils.normalize_features(features)

        # 构建邻接矩阵（使用供应链 + 同行业边）
        adj = np.zeros((len(stock_list), len(stock_list)), dtype=np.float32)
        for etype in ("supply_chain", "same_industry"):
            for src, dst in graph_data["edges"].get(etype, []):
                if src < len(stock_list) and dst < len(stock_list):
                    adj[src, dst] = 1.0
                    adj[dst, src] = 1.0

        return self.predict(features, adj, stock_list)


# ── Checkpoint Loading Utility ──────────────────────────────────────────────

def load_gnn_checkpoint(model_path: str, device: str = 'cpu') -> dict:
    """加载 GNN checkpoint 并返回模型与元数据。

    这是 run_full_period_backtest / run_final_backtest 脚本中
    GNN 模型加载逻辑的模块化封装。支持 GCN 和 GAT 两种架构。

    Args:
        model_path: .pt checkpoint 文件路径
        device: 'cpu' | 'cuda'

    Returns:
        {model, model_type, stock_codes, edge_index, features, hidden_dim, dropout}
        若文件不存在返回空 dict。
    """
    from pathlib import Path

    import torch
    import torch.nn.functional as F

    if not Path(model_path).exists():
        return {}

    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    model_type = ckpt.get('model_type', 'GCN')
    in_dim = len(ckpt['features'])
    hidden = ckpt.get('hidden_dim', 64)
    dropout = ckpt.get('dropout', 0.5)
    stock_codes = ckpt['stock_codes']
    edge_index = ckpt['edge_index'].to(device)

    if model_type == 'GAT':
        try:
            from torch_geometric.nn import GATConv
            heads = ckpt.get('gat_heads', 4)

            class _GATModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
                    self.conv2 = GATConv(hidden * heads, 1, heads=1, dropout=dropout)
                    self._dp = dropout

                def forward(self, x, ei):
                    x = F.dropout(x, p=self._dp, training=False)
                    x = F.elu(self.conv1(x, ei))
                    x = F.dropout(x, p=self._dp, training=False)
                    return self.conv2(x, ei)

            model = _GATModel()
        except ImportError:
            model = None
    else:
        try:
            from torch_geometric.nn import GCNConv

            class _GCNModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(in_dim, hidden)
                    self.conv2 = GCNConv(hidden, 1)
                    self._dp = dropout

                def forward(self, x, ei):
                    x = F.relu(self.conv1(x, ei))
                    x = F.dropout(x, p=self._dp, training=False)
                    return self.conv2(x, ei)

            model = _GCNModel()
        except ImportError:
            model = None

    if model is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

    return {
        'model': model,
        'model_type': model_type,
        'stock_codes': stock_codes,
        'edge_index': edge_index,
        'features': ckpt['features'],
        'hidden_dim': hidden,
        'dropout': dropout,
        'device': device,
    }


def run_gnn_inference(checkpoint: dict, factor_data: dict, device: str = 'cpu') -> dict[str, dict[str, float]]:
    """对 factor_data 的每个日期运行 GNN 推理，返回 {date: {code: score}}。

    Args:
        checkpoint: load_gnn_checkpoint() 的返回值
        factor_data: {date: {code: {factor_name: value}}}
        device: 'cpu' | 'cuda'

    Returns:
        {date: {code: gnn_score}}
    """
    from math import isnan

    import numpy as np
    import torch

    model = checkpoint['model']
    if model is None:
        return {}

    edge_index = checkpoint['edge_index'].to(device)
    stock_codes = checkpoint['stock_codes']
    feature_names = checkpoint['features']
    model.to(device)

    gnn_scores = {}
    for mdate, fv_d in sorted(factor_data.items()):
        feats = np.zeros((len(stock_codes), len(feature_names)), dtype=np.float32)
        for i, code in enumerate(stock_codes):
            if code in fv_d:
                for j, fn in enumerate(feature_names):
                    v = fv_d[code].get(fn)
                    if v is not None and not isnan(v) and abs(v) < 1e8:
                        feats[i, j] = v
        x = torch.from_numpy(feats).float().to(device)
        with torch.no_grad():
            preds = model(x, edge_index).cpu().numpy().ravel()
        gnn_scores[mdate] = {code: float(preds[i]) for i, code in enumerate(stock_codes)}

    return gnn_scores
