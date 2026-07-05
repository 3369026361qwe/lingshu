"""
分层风险平价 (HRP) 优化器 (v4.0).

Marcos Lopez de Prado 算法: 层次聚类 + 递归二等分.
与 Black-Litterman 互补 — BL 适合有主观观点时, HRP 适合纯数据驱动.

Usage:
    from juece.hrp import HRPOptimizer
    result = HRPOptimizer.allocate(returns_matrix)
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class HRPResult:
    weights: dict[int, Decimal]
    cluster_tree: list[tuple[int, int, float, int]]
    risk_contribution: dict[int, Decimal]


class HRPOptimizer:
    """分层风险平价 — 聚类 + 递归二等分."""

    @staticmethod
    def allocate(
        returns_matrix: list[list[Decimal]],
        n_assets: int | None = None,
    ) -> HRPResult:
        """HRP 主算法.

        1. 计算距离矩阵 D[i][j] = sqrt(0.5*(1 - ρ[i][j]))
        2. 层次聚类 (single-linkage)
        3. 递归二等分分配权重
        """
        n = n_assets or len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        data = [[float(v) for v in r[:n_obs]] for r in returns_matrix[:n]]

        # 相关矩阵
        corr = HRPOptimizer._correlation_matrix(data, n, n_obs)

        # 距离矩阵
        dist = [[math.sqrt(0.5 * (1 - max(min(corr[i][j], 1.0), -1.0)))
                 for j in range(n)] for i in range(n)]

        # 方差
        var = [Decimal(str(sum((data[i][t] - sum(data[i]) / n_obs) ** 2
                               for t in range(n_obs)) / max(1, n_obs - 1))) for i in range(n)]

        # 初始: 每个资产一个聚类
        clusters = [{i} for i in range(n)]
        cluster_vars = {i: var[i] for i in range(n)}
        cluster_weights = {i: Decimal("1") for i in range(n)}
        tree = []

        # 层次聚类 (single-linkage)
        while len(clusters) > 1:
            # 找最近的两个聚类
            min_d = float("inf")
            merge = (0, 1)
            for a_idx in range(len(clusters)):
                for b_idx in range(a_idx + 1, len(clusters)):
                    d = HRPOptimizer._cluster_distance(
                        clusters[a_idx], clusters[b_idx], dist
                    )
                    if d < min_d:
                        min_d = d
                        merge = (a_idx, b_idx)

            a, b = merge
            tree.append((a, b, min_d, len(clusters[a]) + len(clusters[b])))

            # 合并
            new_cluster = clusters[a] | clusters[b]
            new_var = cluster_vars[a] + cluster_vars[b]

            # 逆方差权重分配
            inv_var_a = safe_divide(Decimal("1"), cluster_vars[a])
            inv_var_b = safe_divide(Decimal("1"), cluster_vars[b])
            alloc_a = safe_divide(inv_var_a, inv_var_a + inv_var_b)
            alloc_b = Decimal("1") - alloc_a

            # 向下传播
            for member in clusters[a]:
                cluster_weights[member] = alloc_a * cluster_weights[member]
            for member in clusters[b]:
                cluster_weights[member] = alloc_b * cluster_weights[member]

            # 更新列表
            clusters.pop(max(a, b))
            clusters.pop(min(a, b))
            clusters.append(new_cluster)
            cluster_vars[len(clusters) - 1] = new_var

        # 归一化权重
        total_w = sum(cluster_weights.values())
        weights = {i: safe_divide(cluster_weights[i], total_w)
                   if total_w > 0 else Decimal("1") / Decimal(n)
                   for i in range(n)}

        return HRPResult(weights=weights, cluster_tree=tree,
                         risk_contribution={i: Decimal("0") for i in range(n)})

    @staticmethod
    def _correlation_matrix(data, n, n_obs):
        means = [sum(col) / n_obs for col in data]
        stds = [math.sqrt(sum((data[i][t] - means[i]) ** 2
                              for t in range(n_obs)) / max(1, n_obs - 1)) for i in range(n)]
        corr = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if stds[i] > 1e-10 and stds[j] > 1e-10:
                    c = sum((data[i][t] - means[i]) * (data[j][t] - means[j])
                            for t in range(n_obs)) / (n_obs - 1)
                    corr[i][j] = c / (stds[i] * stds[j])
        return corr

    @staticmethod
    def _cluster_distance(cluster_a, cluster_b, dist):
        md = float("inf")
        for i in cluster_a:
            for j in cluster_b:
                if dist[i][j] < md:
                    md = dist[i][j]
        return md
