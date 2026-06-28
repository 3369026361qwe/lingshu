"""
波动率因子: 历史波动率、下行波动率、Beta、VaR。

数据来源: 日线行情数据。
"""

from decimal import Decimal

from yinzi.factor_base import FactorBase, FactorCategory


def _daily_returns(daily_data: dict) -> list[Decimal]:
    """从日线数据计算日收益率序列。"""
    sorted_dates = sorted(daily_data.keys())
    returns = []
    for i in range(1, len(sorted_dates)):
        prev_close = daily_data[sorted_dates[i - 1]].get("close")
        curr_close = daily_data[sorted_dates[i]].get("close")
        if prev_close and curr_close and prev_close > 0:
            returns.append(
                (Decimal(str(curr_close)) - Decimal(str(prev_close)))
                / Decimal(str(prev_close))
            )
    return returns


class HistoricalVolFactor(FactorBase):
    """历史波动率 = 日收益率标准差 * sqrt(252) (年化)。方向: -1。"""
    name = "historical_vol"
    category = FactorCategory.VOLATILITY
    description = "历史波动率 (年化)"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        returns = _daily_returns(daily_data)
        if len(returns) < 20:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        daily_vol = variance.sqrt()
        return (daily_vol * Decimal("252").sqrt()).quantize(Decimal("0.0001"))


class DownsideVolFactor(FactorBase):
    """下行波动率 = 仅负收益率的标准差 * sqrt(252)。方向: -1。"""
    name = "downside_vol"
    category = FactorCategory.VOLATILITY
    description = "下行波动率 (年化)"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        returns = _daily_returns(daily_data)
        neg_returns = [r for r in returns if r < 0]
        if len(neg_returns) < 10:
            return None
        mean = sum(neg_returns) / len(neg_returns)
        variance = sum((r - mean) ** 2 for r in neg_returns) / len(neg_returns)
        daily_vol = variance.sqrt()
        return (daily_vol * Decimal("252").sqrt()).quantize(Decimal("0.0001"))


class BetaFactor(FactorBase):
    """Beta = Cov(stock, market) / Var(market)。方向: 0 (中性，Beta 本身无好坏)。"""
    name = "beta"
    category = FactorCategory.VOLATILITY
    description = "Beta 系数"
    direction = 0

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        market_data = kwargs.get("market_data", {})
        if not market_data:
            return None

        stock_returns = _daily_returns(daily_data)
        market_returns = _daily_returns(market_data)
        n = min(len(stock_returns), len(market_returns))
        if n < 60:
            return None

        stock_returns = stock_returns[-n:]
        market_returns = market_returns[-n:]

        stock_mean = sum(stock_returns) / n
        market_mean = sum(market_returns) / n

        cov = sum(
            (stock_returns[i] - stock_mean) * (market_returns[i] - market_mean)
            for i in range(n)
        ) / n
        var_market = sum((r - market_mean) ** 2 for r in market_returns) / n

        if var_market == 0:
            return None
        return (cov / var_market).quantize(Decimal("0.0001"))


class VaRFactor(FactorBase):
    """Historical VaR (95%) = 收益率分布的第 5 百分位数 * sqrt(252)。方向: -1。"""
    name = "var_95"
    category = FactorCategory.VOLATILITY
    description = "历史 VaR (95%)"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        returns = _daily_returns(daily_data)
        if len(returns) < 60:
            return None
        sorted_returns = sorted(returns)
        idx = int(len(sorted_returns) * 0.05)
        var_daily = abs(sorted_returns[max(0, idx)])
        return (var_daily * Decimal("252").sqrt()).quantize(Decimal("0.0001"))
