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

import numpy as np

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

    # ── 多周期 IC 衰减 ─────────────────────────────────

    @staticmethod
    def compute_ic_decay(
        factor_values_by_date: dict[str, dict[str, Decimal]],
        forward_returns_by_horizon: dict[int, dict[str, dict[str, Decimal]]],
        horizons: list[int] = (5, 10, 20, 40, 60),
    ) -> dict[int, dict]:
        """多周期 IC 衰减分析。

        计算因子在不同前瞻期下的 Rank IC，评估因子预测力的时间衰减特征。

        Args:
            factor_values_by_date: {date: {code: factor_value}}
            forward_returns_by_horizon: {horizon: {date: {code: return}}}
            horizons: 前瞻期列表

        Returns:
            {horizon: {mean_ic, std_ic, ir, n_periods, ic_series}}
        """
        result = {}
        for h in horizons:
            ic_list = []
            common_dates = set(factor_values_by_date) & set(forward_returns_by_horizon.get(h, {}))
            for d in sorted(common_dates):
                fv = factor_values_by_date[d]
                fr = forward_returns_by_horizon[h][d]
                ic = FactorValidator.compute_rank_ic(fv, fr)
                if ic is not None:
                    ic_list.append(ic)

            if len(ic_list) >= 3:
                ic_vals = [float(x) for x in ic_list]
                result[h] = {
                    "mean_ic": round(sum(ic_vals) / len(ic_vals), 6),
                    "std_ic": round(np.std(ic_vals) if len(ic_vals) > 1 else 0, 6),
                    "ir": round((sum(ic_vals) / len(ic_vals)) / (np.std(ic_vals) if np.std(ic_vals) > 0 else 1), 4),
                    "n_periods": len(ic_list),
                    "ic_series": [float(x) for x in ic_list],
                }
            else:
                result[h] = {"mean_ic": 0, "std_ic": 0, "ir": 0, "n_periods": 0, "ic_series": []}
        return result

    # ── 因子自相关（换手率代理）─────────────────────────

    @staticmethod
    def compute_factor_autocorr(
        factor_values_by_date: dict[str, dict[str, Decimal]],
    ) -> float:
        """计算相邻期因子值的秩自相关系数（越低越好，高自相关=低换手率风险）。

        Returns:
            平均秩自相关系数，-1 到 1 之间。
        """
        import numpy as np
        dates = sorted(factor_values_by_date.keys())
        if len(dates) < 3:
            return 0.0

        cors = []
        for i in range(1, len(dates)):
            prev = factor_values_by_date[dates[i - 1]]
            curr = factor_values_by_date[dates[i]]
            common = set(prev) & set(curr)
            if len(common) < 30:
                continue
            # Rank transformation within each period
            prev_sorted = sorted(common, key=lambda c: prev[c])
            curr_sorted = sorted(common, key=lambda c: curr[c])
            prev_ranks = {c: r for r, c in enumerate(prev_sorted)}
            curr_ranks = {c: r for r, c in enumerate(curr_sorted)}
            r = np.corrcoef(
                [prev_ranks[c] for c in common],
                [curr_ranks[c] for c in common],
            )[0, 1]
            if not np.isnan(r):
                cors.append(r)

        return float(np.mean(cors)) if cors else 0.0

    # ── 滚动 IC 稳定性 ─────────────────────────────────

    @staticmethod
    def ic_stability(
        ic_series: list[float],
        window: int = 60,
    ) -> dict:
        """滚动窗口 IC 稳定性分析。

        Returns:
            {mean_rolling_ic, std_rolling_ic, max_ic, min_ic, ic_series_rolling}
        """
        if len(ic_series) < window:
            return {"mean_rolling_ic": 0, "std_rolling_ic": 0, "max_ic": 0, "min_ic": 0, "ic_series_rolling": []}

        rolling = []
        for i in range(window, len(ic_series) + 1):
            w = ic_series[i - window:i]
            rolling.append(round(sum(w) / len(w), 6))

        return {
            "mean_rolling_ic": round(np.mean(rolling), 6),
            "std_rolling_ic": round(np.std(rolling), 6),
            "max_ic": round(max(rolling), 6),
            "min_ic": round(min(rolling), 6),
            "ic_series_rolling": rolling,
        }

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
