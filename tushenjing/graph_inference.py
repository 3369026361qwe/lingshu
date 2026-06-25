"""
图神经网络推理引擎。

对训练好的 GNN 模型执行推理，产出信息增强的因子得分。
"""

import logging
import time as _time
from typing import Optional

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
        feature_order: Optional[list[str]] = None,
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
