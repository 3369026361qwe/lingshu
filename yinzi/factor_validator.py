"""
因子有效性检验器。

指标:
    IC (Information Coefficient)     — Rank IC: 因子值与未来收益的秩相关系数
    IR (Information Ratio)           — IC 均值 / IC 标准差
    分层回测                          — 按因子值分10组，计算各组平均收益

Usage:
    validator = FactorValidator()
    ic = validator.compute_rank_ic(factor_values, forward_returns)
    ir = validator.compute_ir(ic_series)
    layers = validator.layered_backtest(factor_values, forward_returns, n_groups=10)
"""

import logging
from decimal import Decimal
from typing import Optional

from yinzi.metrics import factor_ic as factor_ic_gauge, factor_ir as factor_ir_gauge, factor_coverage

_logger = logging.getLogger(__name__)


class FactorValidator:
    """因子有效性检验器。"""

    # ── Rank IC ────────────────────────────────────────

    @staticmethod
    def compute_rank_ic(
        factor_values: dict[str, Decimal],   # {code: factor_value}
        forward_returns: dict[str, Decimal], # {code: next_period_return}
    ) -> Optional[Decimal]:
        """计算 Rank IC (Spearman 秩相关系数)。

        IC > 0.02 为有效因子, IC > 0.05 为强有效。
        """
        common_codes = set(factor_values) & set(forward_returns)
        if len(common_codes) < 30:
            return None

        # 提取值对
        pairs = [(factor_values[c], forward_returns[c]) for c in common_codes]
        pairs = [(fv, fr) for fv, fr in pairs if fv is not None]

        if len(pairs) < 30:
            return None

        # 排序 → 秩
        codes = [p[0] for p in pairs]  # 占位，只用于区分
        fv_sorted = sorted(enumerate(pairs), key=lambda x: x[1][0])
        fr_sorted = sorted(enumerate(pairs), key=lambda x: x[1][1])

        fv_ranks = {i: rank + 1 for rank, (i, _) in enumerate(fv_sorted)}
        fr_ranks = {i: rank + 1 for rank, (i, _) in enumerate(fr_sorted)}

        n = len(pairs)
        # Spearman: 1 - 6 * Σd² / (n(n²-1))
        d_squared_sum = sum(
            (fv_ranks[i] - fr_ranks[i]) ** 2 for i in range(n)
        )
        ic = Decimal("1") - Decimal("6") * Decimal(str(d_squared_sum)) / (
            Decimal(str(n)) * (Decimal(str(n ** 2)) - Decimal("1"))
        )
        return ic.quantize(Decimal("0.0001"))

    # ── IR ─────────────────────────────────────────────

    @staticmethod
    def compute_ir(ic_series: list[Decimal]) -> Optional[Decimal]:
        """IR = IC 均值 / IC 标准差。IR > 0.5 为良好，IR > 1.0 为优秀。"""
        if len(ic_series) < 12:
            return None
        mean_ic = sum(ic_series) / len(ic_series)
        variance = sum((ic - mean_ic) ** 2 for ic in ic_series) / len(ic_series)
        std_ic = variance.sqrt()
        if std_ic == 0:
            return None
        return (mean_ic / std_ic).quantize(Decimal("0.0001"))

    # ── 分层回测 ───────────────────────────────────────

    @staticmethod
    def layered_backtest(
        factor_values: dict[str, Decimal],
        forward_returns: dict[str, Decimal],
        n_groups: int = 10,
    ) -> list[dict]:
        """分层回测: 按因子值分 N 组，计算各组平均收益。

        Returns:
            [{group: 1, count: 50, avg_return: 0.015, ...}, ...]
        """
        common_codes = set(factor_values) & set(forward_returns)
        pairs = [(c, factor_values[c], forward_returns[c]) for c in common_codes]
        pairs = [(c, fv, fr) for c, fv, fr in pairs if fv is not None]

        if len(pairs) < n_groups * 3:
            return []

        # 按因子值排序
        pairs.sort(key=lambda x: x[1])

        group_size = len(pairs) // n_groups
        results = []

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else len(pairs)
            group_pairs = pairs[start:end]
            if not group_pairs:
                continue
            avg_return = sum(p[2] for p in group_pairs) / len(group_pairs)
            avg_factor = sum(p[1] for p in group_pairs) / len(group_pairs)
            results.append({
                "group": g + 1,
                "count": len(group_pairs),
                "avg_return": avg_return.quantize(Decimal("0.000001")),
                "avg_factor": avg_factor.quantize(Decimal("0.01")),
            })

        return results

    # ── P1-3: NDCG@k + ranking_metric ──────────────────

    @staticmethod
    def compute_ndcg(factor_values: dict[str, Decimal], forward_returns: dict[str, Decimal], k: int = 30) -> Optional[Decimal]:
        """NDCG@k排序质量指标。"""
        common = set(factor_values) & set(forward_returns)
        if len(common) < k: return None
        pairs = [(c, float(factor_values[c]), float(forward_returns[c])) for c in common]
        pairs = [(c, fv, fr) for c, fv, fr in pairs if fv is not None]
        if len(pairs) < k: return None

        pred = sorted(pairs, key=lambda x: x[1], reverse=True)[:k]
        ideal = sorted(pairs, key=lambda x: x[2], reverse=True)[:k]

        def dcg(items, kk):
            return sum((2.0 ** gain - 1.0) / (__import__('math').log2(i + 2)) for i, (_, _, gain) in enumerate(items[:kk]) if gain > 0)

        import math
        dcgv = sum((2.0 ** gain - 1.0) / math.log2(i + 2) for i, (_, _, gain) in enumerate(pred[:k]) if gain > 0)
        idcgv = sum((2.0 ** gain - 1.0) / math.log2(i + 2) for i, (_, _, gain) in enumerate(ideal[:k]) if gain > 0)
        if idcgv == 0: return None
        return Decimal(str(round(dcgv / idcgv, 6)))

    @staticmethod
    def compute_ranking_metric(factor_values: dict[str, Decimal], forward_returns: dict[str, Decimal], k: int = 5) -> Optional[Decimal]:
        """THU-BDC2026同款:归一化Top-K排序质量=(pred_top_sum-random_sum)/(true_top_sum-random_sum)。"""
        common = set(factor_values) & set(forward_returns)
        pairs = [(c, float(factor_values[c]), float(forward_returns[c])) for c in common]
        if len(pairs) < k: return None
        pairs.sort(key=lambda x: x[1], reverse=True); ps = sum(p[2] for p in pairs[:k])
        pairs.sort(key=lambda x: x[2], reverse=True); ms = sum(p[2] for p in pairs[:k])
        rs = k * sum(p[2] for p in pairs) / len(pairs); d = ms - rs
        if abs(d) < 1e-12: return Decimal("0")
        return Decimal(str(round((ps - rs) / d, 6)))

    # ── 综合检验 ───────────────────────────────────────

    @staticmethod
    def validate(
        factor_name: str,
        factor_values: dict[str, Decimal],
        forward_returns: dict[str, Decimal],
        ic_series: Optional[list[Decimal]] = None,
    ) -> dict:
        """一站式因子有效性检验。

        Args:
            factor_name: 因子名
            factor_values: {code: factor_value}
            forward_returns: {code: forward_return}
            ic_series: IC 时间序列（用于计算 IR，建议 >= 12 期）

        Returns:
            {factor_name, ic, ir, n_stocks, n_valid, layers, top_bottom_spread, monotonic}
        """
        ic = FactorValidator.compute_rank_ic(factor_values, forward_returns)
        layers = FactorValidator.layered_backtest(factor_values, forward_returns)

        result = {
            "factor_name": factor_name,
            "ic": ic,
            "ir": None,
            "n_stocks": len(factor_values),
            "n_valid": len(set(factor_values) & set(forward_returns)),
            "layers": layers,
        }

        # P2-4: 计算 IR（需要 IC 时间序列）
        if ic_series and len(ic_series) >= 12:
            result["ir"] = FactorValidator.compute_ir(ic_series)

        if layers and len(layers) >= 2:
            top_return = layers[-1]["avg_return"]
            bottom_return = layers[0]["avg_return"]
            result["top_bottom_spread"] = (top_return - bottom_return).quantize(Decimal("0.000001"))
            result["monotonic"] = top_return > bottom_return

        # 上报 Prometheus 指标
        if ic is not None:
            factor_ic_gauge.labels(factor_name=factor_name).set(float(ic))
        if result["ir"] is not None:
            factor_ir_gauge.labels(factor_name=factor_name).set(float(result["ir"]))
        coverage = result["n_valid"] / max(result["n_stocks"], 1)
        factor_coverage.labels(factor_name=factor_name).set(coverage)

        return result
