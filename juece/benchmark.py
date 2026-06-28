"""
基准比较器 — 策略表现 vs 沪深300 / 中证500。
"""

from decimal import Decimal


class Benchmark:
    """基准比较器。"""

    BENCHMARKS = {
        "hs300": "沪深300",
        "zz500": "中证500",
        "cyb": "创业板指",
    }

    # ── 收益比较 ────────────────────────────────────────

    @staticmethod
    def compare_returns(
        strategy_return: Decimal,      # 策略累计收益
        benchmark_returns: dict[str, Decimal],  # {hs300: return, zz500: return}
    ) -> dict:
        """计算超额收益。

        Returns:
            {benchmark_name: {benchmark_return, strategy_return, excess_return, alpha_ratio}}
        """
        result = {}
        for bm_key, bm_return in benchmark_returns.items():
            bm_name = Benchmark.BENCHMARKS.get(bm_key, bm_key)
            excess = strategy_return - bm_return
            result[bm_key] = {
                "benchmark": bm_name,
                "benchmark_return": bm_return.quantize(Decimal("0.000001")),
                "strategy_return": strategy_return.quantize(Decimal("0.000001")),
                "excess_return": excess.quantize(Decimal("0.000001")),
                "win": excess > 0,
            }
        return result

    # ── 风险调整 ────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(
        daily_returns: list[Decimal],
        risk_free_rate: Decimal = Decimal("0.03"),
    ) -> Decimal | None:
        """年化夏普比率。

        Sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
        if len(daily_returns) < 20:
            return None

        n = Decimal(len(daily_returns))
        mean_daily = sum(daily_returns) / n

        var = sum((r - mean_daily) ** 2 for r in daily_returns) / n
        std_daily = var.sqrt()
        if std_daily == 0:
            return None

        # 年化
        ann_return = mean_daily * Decimal("252")
        ann_std = std_daily * Decimal("252").sqrt()

        return ((ann_return - risk_free_rate) / ann_std).quantize(Decimal("0.0001"))

    @staticmethod
    def max_drawdown(equity_curve: list[Decimal]) -> Decimal | None:
        """最大回撤。"""
        if not equity_curve:
            return None

        peak = equity_curve[0]
        max_dd = Decimal("0")

        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else Decimal("0")
            if dd > max_dd:
                max_dd = dd

        return max_dd.quantize(Decimal("0.0001"))

    @staticmethod
    def win_rate(daily_returns: list[Decimal]) -> Decimal | None:
        """日胜率。"""
        if not daily_returns:
            return None
        wins = sum(1 for r in daily_returns if r > 0)
        return (Decimal(str(wins)) / Decimal(str(len(daily_returns)))).quantize(Decimal("0.0001"))

    @staticmethod
    def ranking_metric(predictions: dict[str, Decimal], actual_returns: dict[str, Decimal], k: int = 5) -> Decimal | None:
        """P2-5: THU-BDC2026归一化Top-K排序质量。"""
        common = set(predictions) & set(actual_returns)
        pairs = [(c, float(predictions[c]), float(actual_returns[c])) for c in common]
        if len(pairs) < k:
            return None
        pairs.sort(key=lambda x: x[1], reverse=True)
        ps = sum(p[2] for p in pairs[:k])
        pairs.sort(key=lambda x: x[2], reverse=True)
        ms = sum(p[2] for p in pairs[:k])
        rs = k * sum(p[2] for p in pairs) / len(pairs)
        d = ms - rs
        if abs(d) < 1e-12:
            return Decimal("0")
        return Decimal(str(round((ps - rs) / d, 6)))
