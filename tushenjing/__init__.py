"""
tushenjing — 图神经网络引擎 ★ 核心创新

构建 A 股异构图 (5000+ 节点, 4 种边类型) + GAT 模型信息传播 + 推理 + 动态更新。

边类型:
    产业链上下游 — 从财报/供应链数据提取
    同行业关系     — 申万行业分类
    共同持仓关系   — 基金重仓股交叉
    概念板块归属   — Wind/同花顺概念

模型: GCN → GAT → HGT (可选)
输入: 股票特征向量 (因子值 + Agent 评分)
输出: 信息增强因子得分

Usage:
    from tushenjing import GraphBuilder, GATModel, GraphInference
"""

from tushenjing.gnn_model import GATModel, GCNModel, create_model
from tushenjing.graph_builder import GraphBuilder
from tushenjing.graph_utils import GraphUtils

__all__ = [
    "GraphUtils", "GraphBuilder",
    "GCNModel", "GATModel", "create_model",
]
