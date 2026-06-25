"""
图结构动态更新器。

增量更新节点特征和边关系，避免每次全量重建。
"""

import logging
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)


class GraphUpdater:
    """图结构动态更新器。"""

    def __init__(self, graph_data: dict):
        self._graph = graph_data
        self._node_to_idx: dict[str, int] = graph_data.get("node_to_idx", {})
        self._edges: dict[str, list[tuple[int, int]]] = graph_data.get("edges", {})

    # ── 节点更新 ────────────────────────────────────────

    def add_nodes(self, new_codes: list[str]) -> int:
        """添加新股票节点。

        Returns:
            新增节点数
        """
        added = 0
        nodes = list(self._graph["nodes"])
        for code in new_codes:
            if code not in self._node_to_idx:
                self._node_to_idx[code] = len(nodes)
                nodes.append(code)
                added += 1
        self._graph["nodes"] = nodes
        self._graph["num_nodes"] = len(nodes)
        return added

    def remove_nodes(self, codes: list[str]) -> int:
        """移除退市/停牌股票节点。"""
        removed = 0
        for code in codes:
            if code in self._node_to_idx:
                del self._node_to_idx[code]
                removed += 1
        # 重建索引
        nodes = [c for c in self._graph["nodes"] if c in self._node_to_idx]
        self._node_to_idx = {c: i for i, c in enumerate(nodes)}
        self._graph["nodes"] = nodes
        self._graph["num_nodes"] = len(nodes)
        return removed

    # ── 边更新 ──────────────────────────────────────────

    def add_edges(self, edge_type: str, new_edges: list[tuple[str, str]]) -> int:
        """添加新边。

        Args:
            edge_type: 边类型 (supply_chain/same_industry/co_holding/concept)
            new_edges: [(code1, code2)]

        Returns:
            新增边数
        """
        added = 0
        for c1, c2 in new_edges:
            i1 = self._node_to_idx.get(c1)
            i2 = self._node_to_idx.get(c2)
            if i1 is not None and i2 is not None:
                self._edges.setdefault(edge_type, []).append((i1, i2))
                added += 1
        self._graph["num_edges"] = sum(len(e) for e in self._edges.values())
        return added

    def remove_edges(self, edge_type: str, old_edges: list[tuple[str, str]]) -> int:
        """移除边。"""
        if edge_type not in self._edges:
            return 0
        to_remove = set()
        for c1, c2 in old_edges:
            i1 = self._node_to_idx.get(c1)
            i2 = self._node_to_idx.get(c2)
            if i1 is not None and i2 is not None:
                to_remove.add((i1, i2))
                to_remove.add((i2, i1))

        self._edges[edge_type] = [e for e in self._edges[edge_type] if (e[0], e[1]) not in to_remove]
        self._graph["num_edges"] = sum(len(e) for e in self._edges.values())
        return len(to_remove)

    # ── 特征更新 ────────────────────────────────────────

    @staticmethod
    def update_features(
        graph_data: dict,
        factor_data: dict[str, dict],
        feature_order: Optional[list[str]] = None,
    ) -> np.ndarray:
        """更新节点特征矩阵。

        Returns:
            更新后的特征矩阵 [N, F]
        """
        from tushenjing.graph_utils import GraphUtils
        features, _ = GraphUtils.build_feature_matrix(
            graph_data["nodes"], factor_data, feature_order
        )
        return GraphUtils.normalize_features(features)

    @property
    def graph_data(self) -> dict:
        return self._graph
