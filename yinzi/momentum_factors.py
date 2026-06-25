"""
动量因子: 1月/3月/6月/12-1月动量、换手率动量。

数据来源: 日线行情数据 (daily_data)。
"""

from decimal import Decimal
from typing import Optional

from yinzi.factor_base import FactorBase, FactorCategory, FactorResult


class _MomentumBase(FactorBase):
    """动量因子基类。"""
    category = FactorCategory.MOMENTUM
    direction = 1
    _lookback_days: int = 21

    def supports_vectorized(self) -> bool: return True

    def _momentum(self, daily_data: dict, lookback_days: int) -> Optional[Decimal]:
        """计算回溯期收益率 = (最新收盘价 - N日前收盘价) / N日前收盘价。"""
        if not daily_data or len(daily_data) < lookback_days:
            return None
        sorted_dates = sorted(daily_data.keys())[-lookback_days:]
        if len(sorted_dates) < lookback_days:
            return None
        start_price = daily_data[sorted_dates[0]].get("close")
        end_price = daily_data[sorted_dates[-1]].get("close")
        if start_price and end_price and start_price > 0:
            s = Decimal(str(start_price))
            e = Decimal(str(end_price))
            return (e - s) / s
        return None

    def compute_vectorized(self, stock_list, daily_data_map, financial_data_map=None, **kwargs):
        results = []
        for code in stock_list:
            daily = daily_data_map.get(code, {})
            value = self._momentum(daily, self._lookback_days)
            if value is not None:
                results.append(FactorResult(code=code, factor_name=self.name, category=self.category, raw_value=value))
        return results


class Momentum1MFactor(_MomentumBase):
    """1 月动量 (约 21 个交易日)。"""
    name = "momentum_1m"
    description = "1月动量"
    _lookback_days = 21

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        return self._momentum(daily_data, self._lookback_days)


class Momentum3MFactor(_MomentumBase):
    """3 月动量 (约 63 个交易日)。"""
    name = "momentum_3m"
    description = "3月动量"
    _lookback_days = 63

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        return self._momentum(daily_data, self._lookback_days)


class Momentum6MFactor(_MomentumBase):
    """6 月动量 (约 126 个交易日)。"""
    name = "momentum_6m"
    description = "6月动量"
    _lookback_days = 126

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        return self._momentum(daily_data, self._lookback_days)


class Momentum12M1MFactor(_MomentumBase):
    """12-1 月动量 (剔除最近 1 月，看 2-12 月趋势)。"""
    name = "momentum_12m1m"
    description = "12-1月动量"

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if not daily_data or len(daily_data) < 252:
            return None
        sorted_dates = sorted(daily_data.keys())
        # 12个月前 → 1个月前
        start_date = sorted_dates[-252] if len(sorted_dates) >= 252 else sorted_dates[0]
        end_date = sorted_dates[-22] if len(sorted_dates) >= 22 else sorted_dates[-1]
        start_price = daily_data[start_date].get("close")
        end_price = daily_data[end_date].get("close")
        if start_price and end_price and start_price > 0:
            s = Decimal(str(start_price))
            e = Decimal(str(end_price))
            return (e - s) / s
        return None


class TurnoverMomentumFactor(FactorBase):
    """换手率动量: 近期换手率均值的环比变化。方向: 0 (中性，异常换手需结合方向判断)。"""

    name = "turnover_momentum"
    category = FactorCategory.MOMENTUM
    description = "换手率动量"
    direction = 0

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if not daily_data or len(daily_data) < 42:
            return None
        sorted_dates = sorted(daily_data.keys())
        recent = sorted_dates[-21:]   # 近 1 月
        prior = sorted_dates[-42:-21] # 前 1 月

        recent_avg = self._avg_turnover(daily_data, recent)
        prior_avg = self._avg_turnover(daily_data, prior)
        if recent_avg and prior_avg and prior_avg > 0:
            return (recent_avg - prior_avg) / prior_avg
        return None

    @staticmethod
    def _avg_turnover(daily_data: dict, dates: list) -> Optional[Decimal]:
        values = []
        for d in dates:
            tr = daily_data[d].get("turnover_rate")
            if tr is not None:
                values.append(Decimal(str(tr)))
        if not values:
            return None
        return sum(values) / len(values)
