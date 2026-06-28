"""
A股产业链图构建器。

从 shuju 数据层提取 4 种边类型:
    1. 产业链上下游 — 基于申万行业上下游关系
    2. 同行业关系     — 同一申万一级行业
    3. 共同持仓关系   — 基金重仓交叉持有
    4. 概念板块归属   — 同花顺/Wind 概念板块

输出:
    - 节点列表 (5000+ A 股)
    - 边列表 (4 种类型, 12000+ 边)
    - 节点特征矩阵 [N, F]
"""

import logging
import random
import time

import numpy as np

from tushenjing.metrics import graph_build_duration, graph_edges_total, graph_nodes_total

_logger = logging.getLogger(__name__)
random.seed(42)

# 申万行业上下游关系（简化版）
_INDUSTRY_SUPPLY_CHAIN: dict[str, list[str]] = {
    "石油石化": ["基础化工", "交通运输"],
    "煤炭": ["钢铁", "公用事业"],
    "有色金属": ["电子", "电力设备", "汽车"],
    "基础化工": ["医药生物", "农林牧渔", "纺织服饰"],
    "钢铁": ["汽车", "机械设备", "建筑装饰"],
    "电子": ["计算机", "通信", "传媒"],
    "电力设备": ["公用事业", "汽车", "机械设备"],
    "汽车": ["交通运输", "商贸零售"],
    "医药生物": ["美容护理", "社会服务"],
    "食品饮料": ["农林牧渔", "商贸零售"],
    "农林牧渔": ["食品饮料", "纺织服饰"],
    "机械设备": ["汽车", "建筑装饰", "国防军工"],
    "建筑装饰": ["房地产", "建筑材料"],
    "房地产": ["银行", "家用电器"],
    "银行": ["非银金融", "房地产"],
    "计算机": ["电子", "通信", "传媒"],
    "通信": ["传媒", "计算机"],
    "交通运输": ["商贸零售", "汽车"],
}


class GraphBuilder:
    """A 股异构图构建器。"""

    def __init__(self):
        self._nodes: list[str] = []                # 股票代码列表
        self._node_to_idx: dict[str, int] = {}     # code → index
        self._industry_map: dict[str, str] = {}     # code → sw_level1
        self._edges: dict[str, list[tuple[int, int]]] = {
            "supply_chain": [],      # 产业链上下游
            "same_industry": [],     # 同行业
            "co_holding": [],        # 共同持仓
            "concept": [],           # 概念板块
        }

    # ── 构建流程 ────────────────────────────────────────

    def build(
        self,
        stock_list: list[str],
        industry_map: dict[str, str] | None = None,
        fund_holdings: dict[str, list[str]] | None = None,
        concept_map: dict[str, list[str]] | None = None,
    ) -> dict:
        """构建完整异构图。

        Args:
            stock_list: A 股股票列表
            industry_map: {code: sw_level1}
            fund_holdings: {fund_id: [code1, code2, ...]}
            concept_map: {concept_name: [codes]}

        Returns:
            {nodes, node_to_idx, edges, industry_map, num_nodes, num_edges}
        """
        t0 = time.perf_counter()
        self._nodes = sorted(stock_list)
        self._node_to_idx = {code: i for i, code in enumerate(self._nodes)}
        self._industry_map = industry_map or {}

        n = len(self._nodes)
        _logger.info("Building graph with %d nodes", n)

        # 边类型 1: 产业链上下游
        if self._industry_map:
            self._build_supply_chain_edges()
            _logger.info("  supply_chain edges: %d", len(self._edges["supply_chain"]))

        # 边类型 2: 同行业 (P1-2: k-NN采样)
        if self._industry_map:
            self._build_same_industry_edges()
            _logger.info("  same_industry edges: %d", len(self._edges["same_industry"]))

        # 边类型 3: 共同持仓
        if fund_holdings:
            self._build_co_holding_edges(fund_holdings)
            _logger.info("  co_holding edges: %d", len(self._edges["co_holding"]))

        # 边类型 4: 概念板块
        if concept_map:
            self._build_concept_edges(concept_map)
            _logger.info("  concept edges: %d", len(self._edges["concept"]))

        # P2-5: 边去重
        self._deduplicate_edges()

        total_edges = sum(len(e) for e in self._edges.values())
        _logger.info("Graph built: %d nodes, %d total edges", n, total_edges)

        # P0-1: Prometheus 指标
        graph_build_duration.observe(time.perf_counter() - t0)
        graph_nodes_total.set(n)
        for etype, edges in self._edges.items():
            graph_edges_total.labels(edge_type=etype).set(len(edges))

        return {
            "nodes": self._nodes,
            "node_to_idx": self._node_to_idx,
            "edges": self._edges,
            "industry_map": self._industry_map,
            "num_nodes": n,
            "num_edges": total_edges,
        }

    # ── 边类型构建 ──────────────────────────────────────

    def _build_supply_chain_edges(self) -> None:
        """产业链上下游边：上游行业的所有股票 → 下游行业的所有股票。"""
        # 按行业分组
        industry_codes: dict[str, list[str]] = {}
        for code, ind in self._industry_map.items():
            if code in self._node_to_idx:
                industry_codes.setdefault(ind, []).append(code)

        for upstream, downstreams in _INDUSTRY_SUPPLY_CHAIN.items():
            ups = industry_codes.get(upstream, [])
            for down_name in downstreams:
                downs = industry_codes.get(down_name, [])
                for u in ups:
                    ui = self._node_to_idx[u]
                    for d in downs:
                        di = self._node_to_idx[d]
                        self._edges["supply_chain"].append((ui, di))

    def _build_same_industry_edges(self, max_neighbors: int = 20) -> None:
        """P1-2: 同行业边 — k-NN采样代替全连接，避免大板块爆炸。"""
        industry_codes: dict[str, list[str]] = {}
        for code, ind in self._industry_map.items():
            if code in self._node_to_idx:
                industry_codes.setdefault(ind, []).append(code)

        for codes in industry_codes.values():
            n = len(codes)
            if n <= max_neighbors + 1:
                for i in range(n):
                    for j in range(i + 1, n):
                        self._edges["same_industry"].append(
                            (self._node_to_idx[codes[i]], self._node_to_idx[codes[j]])
                        )
            else:
                for i in range(n):
                    others = list(range(n))
                    others.remove(i)
                    sampled = random.sample(others, min(max_neighbors, len(others)))
                    for j in sampled:
                        self._edges["same_industry"].append(
                            (self._node_to_idx[codes[i]], self._node_to_idx[codes[j]])
                        )

    def _deduplicate_edges(self) -> None:
        """P2-5: 每种边类型内部去重。"""
        for etype in self._edges:
            seen = set()
            unique = []
            for edge in self._edges[etype]:
                key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                if key not in seen:
                    seen.add(key)
                    unique.append(edge)
            self._edges[etype] = unique

    def _build_co_holding_edges(self, fund_holdings: dict[str, list[str]]) -> None:
        """共同持仓边：同一基金持有的股票之间建立边。"""
        for _fund_id, codes in fund_holdings.items():
            valid = [c for c in codes if c in self._node_to_idx]
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    ui = self._node_to_idx[valid[i]]
                    uj = self._node_to_idx[valid[j]]
                    self._edges["co_holding"].append((ui, uj))

    def _build_concept_edges(self, concept_map: dict[str, list[str]]) -> None:
        """概念板块边：同一概念板块内的全连接。"""
        for _concept_name, codes in concept_map.items():
            valid = [c for c in codes if c in self._node_to_idx]
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    ui = self._node_to_idx[valid[i]]
                    uj = self._node_to_idx[valid[j]]
                    self._edges["concept"].append((ui, uj))

    # ── 导出 ────────────────────────────────────────────

    def to_adjacency_matrix(self, edge_types: list[str] | None = None) -> np.ndarray:
        """将指定边类型合并为单一邻接矩阵。"""
        if edge_types is None:
            edge_types = list(self._edges.keys())

        adj = np.zeros((len(self._nodes), len(self._nodes)), dtype=np.float32)
        for etype in edge_types:
            for src, dst in self._edges.get(etype, []):
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0  # 对称无向图
        return adj

    def to_pyg_data(self, features: np.ndarray) -> dict:
        """导出为 PyG 兼容的 Data 格式。

        Returns:
            {x, edge_index, edge_type, ...} 字典（非 PyG Data 对象，无需 PyG 导入）
        """
        # 合并所有边类型
        all_edges = []
        edge_type_list = []
        for etype_idx, (_etype_name, edges) in enumerate(self._edges.items()):
            for src, dst in edges:
                all_edges.append([src, dst])
                edge_type_list.append(etype_idx)

        edge_index = np.array(all_edges).T if all_edges else np.zeros((2, 0), dtype=np.int64)

        return {
            "x": features,
            "edge_index": edge_index,
            "edge_type": np.array(edge_type_list, dtype=np.int64) if edge_type_list else None,
            "num_nodes": len(self._nodes),
            "num_edges": len(all_edges),
            "num_edge_types": len(self._edges),
        }
