"""
企业行为调整器 — 复权因子 / 除权除息 (v4.0).

处理送股、转增、分红、配股对价格的影响,
确保回测使用可比较的调整后价格。

Usage:
    from shuju.corporate_action import CorporateActionAdjuster
    adj_prices = CorporateActionAdjuster.adjust_prices(raw_prices, factors)
"""

from decimal import Decimal

from shuju.utils import safe_divide


class CorporateActionAdjuster:
    """企业行为调整器.

    当前为 stub — 完整实现需要股利/拆股数据库。
    """

    @staticmethod
    def adjust_prices(
        raw_prices: list[Decimal],
        adjustment_factors: list[Decimal],
    ) -> list[Decimal]:
        """后向复权: adj_price[t] = raw_price[t] * cum_factor[t].

        cum_factor[t] = Π_{i=t}^{T} factor[i].
        """
        n = min(len(raw_prices), len(adjustment_factors))
        if n == 0:
            return list(raw_prices)

        # 后向累计因子
        cum = Decimal("1")
        cum_factors = [Decimal("1")] * n
        for t in range(n - 1, -1, -1):
            cum *= adjustment_factors[t] if adjustment_factors[t] > 0 else Decimal("1")
            cum_factors[t] = cum

        return [raw_prices[t] * cum_factors[t] for t in range(n)]

    @staticmethod
    def build_adjustment_factors(
        dividends: list[tuple[str, Decimal]],
        splits: list[tuple[str, Decimal]],
        start_date: str,
        end_date: str,
    ) -> list[Decimal]:
        """构建复权因子序列.

        factor = (close_before - dividend) / close_before * split_ratio.
        Stub: 返回全 1 序列.
        """
        # 需要日线收盘价数据库才能正确计算
        return [Decimal("1")]

    @staticmethod
    def adjust_from_close(
        close_before: Decimal,
        dividend_per_share: Decimal,
        split_ratio: Decimal = Decimal("1"),
    ) -> Decimal:
        """单次除权除息调整因子.

        factor = (close_before - dps) / close_before * split_ratio.
        """
        if close_before == 0:
            return Decimal("1")
        return safe_divide(close_before - dividend_per_share, close_before) * split_ratio
