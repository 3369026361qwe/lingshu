"""
多重检验校正 (v4.0).

对 41 个因子的上百次假设检验实施 FWER/FDR 控制，
防止假阳性因子污染信号。

Methods:
    Bonferroni        — FWER, 最保守
    Holm-Bonferroni   — FWER, step-down
    Benjamini-Hochberg — FDR (常用)
    Benjamini-Yekutieli — FDR, 允许任意依赖

Usage:
    from yinzi.multiple_testing import MultipleTestingCorrector
    result = MultipleTestingCorrector.benjamini_hochberg(pvalues)
"""

from dataclasses import dataclass


@dataclass
class MultipleTestingResult:
    method: str
    original_pvalues: list[float]
    adjusted_pvalues: list[float]
    rejected: list[bool]
    n_tests: int
    n_rejected: int
    alpha: float


class MultipleTestingCorrector:
    """多重检验校正 — 完整方法论套件。"""

    @staticmethod
    def bonferroni(
        pvalues: list[float], alpha: float = 0.05
    ) -> MultipleTestingResult:
        """Bonferroni: 拒绝 if p_i < α/m."""
        m = len(pvalues)
        threshold = alpha / m
        rejected = [p < threshold for p in pvalues]
        adjusted = [min(p * m, 1.0) for p in pvalues]
        return MultipleTestingResult(
            method="Bonferroni", original_pvalues=pvalues,
            adjusted_pvalues=adjusted, rejected=rejected,
            n_tests=m, n_rejected=sum(rejected), alpha=alpha,
        )

    @staticmethod
    def holm_bonferroni(
        pvalues: list[float], alpha: float = 0.05
    ) -> MultipleTestingResult:
        """Holm-Bonferroni (step-down): 逐步拒绝."""
        m = len(pvalues)
        indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
        rejected = [False] * m
        adjusted = [1.0] * m

        for k, (idx, p) in enumerate(indexed):
            threshold = alpha / (m - k)
            if p < threshold:
                rejected[idx] = True
            else:
                break  # step-down: 一旦一个不拒绝，后面都不拒绝

        # 调整 p 值
        for k, (idx, p) in enumerate(indexed):
            adjusted[idx] = min(max(
                p * (m - k) for j in range(k + 1)
            ), 1.0)

        return MultipleTestingResult(
            method="Holm-Bonferroni", original_pvalues=pvalues,
            adjusted_pvalues=adjusted, rejected=rejected,
            n_tests=m, n_rejected=sum(rejected), alpha=alpha,
        )

    @staticmethod
    def benjamini_hochberg(
        pvalues: list[float], alpha: float = 0.05
    ) -> MultipleTestingResult:
        """Benjamini-Hochberg FDR 控制.

        Step-up: 拒绝最大的 k 个 p-values 满足 p_{(k)} ≤ k/m · α.
        """
        m = len(pvalues)
        indexed = sorted(enumerate(pvalues), key=lambda x: x[1], reverse=True)
        rejected = [False] * m
        adjusted = [1.0] * m

        # Step-up: 从最大 p-value 开始
        for rank_from_top, (_, p) in enumerate(indexed):
            k = m - rank_from_top
            threshold = k / m * alpha
            if p <= threshold:
                # 这个及所有更小的 p-value 都拒绝
                for rank2, (idx2, _) in enumerate(indexed):
                    if rank2 >= rank_from_top:
                        rejected[idx2] = True
                break

        # BH adjusted p-value: p_BH(i) = min(p(i) * m/i, p_BH(i+1))
        sorted_p = [p for _, p in sorted(enumerate(pvalues), key=lambda x: x[1])]
        adj_sorted = [1.0] * m
        for i in range(m - 1, -1, -1):
            adj_sorted[i] = min(sorted_p[i] * m / (i + 1), 1.0)
            if i < m - 1:
                adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])

        # 映射回原始索引
        idx_to_rank = {idx: r for r, (idx, _) in
                       enumerate(sorted(enumerate(pvalues), key=lambda x: x[1]))}
        for orig_idx, rank in idx_to_rank.items():
            adjusted[orig_idx] = adj_sorted[rank]

        return MultipleTestingResult(
            method="Benjamini-Hochberg", original_pvalues=pvalues,
            adjusted_pvalues=adjusted, rejected=rejected,
            n_tests=m, n_rejected=sum(rejected), alpha=alpha,
        )

    @staticmethod
    def benjamini_yekutieli(
        pvalues: list[float], alpha: float = 0.05
    ) -> MultipleTestingResult:
        """Benjamini-Yekutieli FDR (允许任意依赖).

        与 BH 相同但 α 除以 harmonic sum: α / (Σ 1/i).
        """
        m = len(pvalues)
        harmonic_sum = sum(1.0 / (i + 1) for i in range(m))
        alpha_adjusted = alpha / harmonic_sum

        # 复用 BH 逻辑，使用更保守的 α
        return MultipleTestingCorrector.benjamini_hochberg(
            pvalues, alpha=alpha_adjusted
        )

    @staticmethod
    def compare_all(
        pvalues: list[float], alpha: float = 0.05
    ) -> dict[str, MultipleTestingResult]:
        """运行全部四种方法并对比结果."""
        return {
            "bonferroni": MultipleTestingCorrector.bonferroni(pvalues, alpha),
            "holm": MultipleTestingCorrector.holm_bonferroni(pvalues, alpha),
            "bh": MultipleTestingCorrector.benjamini_hochberg(pvalues, alpha),
            "by": MultipleTestingCorrector.benjamini_yekutieli(pvalues, alpha),
        }
