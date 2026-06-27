"""
图数据处理工具。

提供特征标准化、邻接矩阵构建、边索引转换等基础操作。
不依赖 PyTorch/PyG，使用 numpy/pandas 实现。
"""

import logging
from decimal import Decimal
from typing import Optional

import numpy as np

_logger = logging.getLogger(__name__)


class GraphUtils:
    """图数据处理静态工具集。"""

    # ── 特征处理 ────────────────────────────────────────

    @staticmethod
    def normalize_features(
        features: np.ndarray,          # [N, F]
        method: str = "zscore",
    ) -> np.ndarray:
        """特征标准化。

        Args:
            features: [N 节点, F 特征]
            method: "zscore" | "minmax"

        Returns:
            标准化后的特征矩阵
        """
        if method == "zscore":
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (features - mean) / std
        elif method == "minmax":
            f_min = features.min(axis=0, keepdims=True)
            f_max = features.max(axis=0, keepdims=True)
            denom = f_max - f_min
            denom[denom == 0] = 1.0
            return (features - f_min) / denom
        return features

    @staticmethod
    def build_feature_matrix(
        stock_list: list[str],
        factor_data: dict[str, dict[str, Decimal]],  # {code: {factor: value}}
        feature_order: Optional[list[str]] = None,
        fill_value: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """从因子数据构建特征矩阵。

        Args:
            stock_list: 节点股票列表
            factor_data: {code: {factor_name: value}}
            feature_order: 特征顺序（None 则自动收集）
            fill_value: 缺失值填充

        Returns:
            (features [N, F], feature_names [F])
        """
        if feature_order is None:
            # 自动收集所有因子名
            all_factors = set()
            for factors in factor_data.values():
                all_factors.update(factors.keys())
            feature_order = sorted(all_factors)

        if not feature_order:
            return np.zeros((len(stock_list), 0)), []

        F = len(feature_order)
        N = len(stock_list)
        features = np.full((N, F), fill_value, dtype=np.float32)

        for i, code in enumerate(stock_list):
            factors = factor_data.get(code, {})
            for j, fname in enumerate(feature_order):
                val = factors.get(fname)
                if val is not None:
                    features[i, j] = float(val)

        return features, feature_order

    # ── 邻接矩阵 ────────────────────────────────────────

    @staticmethod
    def edges_to_adjacency(
        edges: list[tuple[int, int]],   # [(src, dst)]
        num_nodes: int,
        symmetric: bool = True,
    ) -> np.ndarray:
        """边列表 → 邻接矩阵 [N, N]。

        Args:
            edges: 边列表（0-indexed 节点索引）
            num_nodes: 节点总数
            symmetric: 是否对称化（无向图）

        Returns:
            邻接矩阵
        """
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src, dst in edges:
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adj[src, dst] = 1.0
                if symmetric:
                    adj[dst, src] = 1.0
        return adj

    @staticmethod
    def adjacency_to_edge_index(adj: np.ndarray) -> np.ndarray:
        """邻接矩阵 → PyG 格式边索引 [2, E]。"""
        rows, cols = np.where(adj > 0)
        return np.stack([rows, cols], axis=0).astype(np.int64)

    @staticmethod
    def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """对称归一化邻接矩阵: D^{-1/2} A D^{-1/2}。"""
        degrees = adj.sum(axis=1)
        d_inv_sqrt = np.zeros_like(degrees)
        mask = degrees > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    # ── 排序损失 ──────────────────────────────────────

    @staticmethod
    def ranking_loss(preds, labels, margin: float = 0.02, max_pairs: int = 500):
        """成对排序损失 (Pairwise Ranking Loss)。

        对随机采样的配对 (i, j)，若 label_i > label_j 但 pred_i < pred_j，
        施加 margin-based hinge loss。用于 GNN 模型训练中的股票排序优化。

        Args:
            preds: 预测分数 tensor [N]
            labels: 真实标签 tensor [N]
            margin: 排序间隔阈值
            max_pairs: 每批最大配对数

        Returns:
            排序损失值 (torch scalar)
        """
        try:
            import torch
            n = len(preds)
            if n < 2:
                return torch.tensor(0.0, device=preds.device)
            idx = torch.randperm(n, device=preds.device)[:max_pairs * 2]
            if len(idx) < 2:
                return torch.tensor(0.0, device=preds.device)
            h = len(idx) // 2
            i_idx = idx[:h]
            j_idx = idx[h:2 * h]
            ld = labels[i_idx] - labels[j_idx]
            pp = ld > 0
            if pp.sum() == 0:
                return torch.tensor(0.0, device=preds.device)
            return torch.clamp(margin - (preds[i_idx][pp] - preds[j_idx][pp]), min=0).mean()
        except ImportError:
            return 0.0

    # ── k-NN 图构建 ────────────────────────────────────

    @staticmethod
    def build_knn_graph(
        stock_codes: list[str],
        factor_by_date: dict[str, dict[str, dict[str, float]]],
        feature_names: list[str],
        industry_map: dict[str, str],
        k_neighbors: int = 10,
        n_dates: int = 200,
    ) -> list[tuple[int, int]]:
        """构建稀疏 k-NN 相关性图 + 行业边。

        基于因子特征向量的相关性距离选择每个节点的 K 个最近邻居，
        并添加最多 5 条同行业边。用于 GNN 模型训练。

        Args:
            stock_codes: 股票代码列表
            factor_by_date: {date: {code: {factor: value}}}
            feature_names: 使用的因子名列表
            industry_map: {code: industry_name}
            k_neighbors: 每个节点的最近邻居数
            n_dates: 用于图构建的日期数（取前 N 天）

        Returns:
            边列表 [(src_idx, dst_idx), ...]
        """
        import numpy as np
        from collections import defaultdict

        n_stocks = len(stock_codes)
        stock_feat_matrix = np.zeros((n_stocks, len(feature_names)), dtype=np.float32)
        count_matrix = np.zeros(n_stocks, dtype=np.int32)

        dates = sorted(factor_by_date.keys())[:n_dates]
        for mdate in dates:
            fv = factor_by_date[mdate]
            for i, code in enumerate(stock_codes):
                if code in fv:
                    for j, fn in enumerate(feature_names):
                        v = fv[code].get(fn)
                        if v is not None and not (hasattr(v, 'isnan') and v.isnan()) and abs(v) < 1e8:
                            stock_feat_matrix[i, j] += float(v)
                            count_matrix[i] += 1

        for i in range(n_stocks):
            if count_matrix[i] > 0:
                stock_feat_matrix[i] /= count_matrix[i]

        mu = stock_feat_matrix.mean(axis=0)
        sg = stock_feat_matrix.std(axis=0) + 1e-12
        feat_norm = (stock_feat_matrix - mu) / sg

        edges = []
        for i in range(n_stocks):
            diff = feat_norm - feat_norm[i]
            dist = np.sqrt((diff ** 2).sum(axis=1))
            dist[i] = 1e9
            neighbors = np.argpartition(dist, k_neighbors)[:k_neighbors]
            for j in neighbors:
                if j != i:
                    edges.append((i, int(j)))
                    edges.append((int(j), i))

        # 行业边（最多 5 条/股）
        ind_groups = defaultdict(list)
        for i, code in enumerate(stock_codes):
            ind = industry_map.get(code, '其他')
            ind_groups[ind].append(i)
        rng = np.random.RandomState(42)
        for members in ind_groups.values():
            for i in members:
                others = [m for m in members if m != i]
                if len(others) > 5:
                    others = list(rng.choice(others, 5, replace=False))
                for j in others:
                    edges.append((i, j))

        return list(set(edges))

    # ── 训练/测试划分 ──────────────────────────────────

    @staticmethod
    def train_test_split_mask(
        num_nodes: int,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成训练/验证节点掩码。"""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(num_nodes)
        split = int(num_nodes * train_ratio)
        train_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[indices[:split]] = True
        test_mask[indices[split:]] = True
        return train_mask, test_mask
