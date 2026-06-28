"""
情绪因子: 成交量比、资金流向、换手率异常、北向资金。

数据来源: 日线行情 + 资金流数据。
"""

from decimal import Decimal

from yinzi.factor_base import FactorBase, FactorCategory


class VolumeRatioFactor(FactorBase):
    """量比 = 近5日均量 / 近20日均量。方向: 0 (需结合价格方向判断)。"""
    name = "volume_ratio"
    category = FactorCategory.SENTIMENT
    description = "成交量比 (5日/20日)"
    direction = 0

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if not daily_data or len(daily_data) < 20:
            return None
        sorted_dates = sorted(daily_data.keys())

        recent_vol = self._avg_volume(daily_data, sorted_dates[-5:])
        base_vol = self._avg_volume(daily_data, sorted_dates[-20:])
        if recent_vol and base_vol and base_vol > 0:
            return (recent_vol / base_vol).quantize(Decimal("0.0001"))
        return None

    @staticmethod
    def _avg_volume(daily_data: dict, dates: list) -> Decimal | None:
        values = []
        for d in dates:
            v = daily_data[d].get("volume")
            if v is not None:
                values.append(Decimal(str(v)))
        return sum(values) / len(values) if values else None


class MoneyFlowFactor(FactorBase):
    """资金流向 = (收盘价 - 开盘价) * 成交量 / 成交额。方向: 1。"""
    name = "money_flow"
    category = FactorCategory.SENTIMENT
    description = "资金流向强度"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if not daily_data:
            return None
        sorted_dates = sorted(daily_data.keys())[-5:]  # 近5日
        total_mf = Decimal("0")
        for d in sorted_dates:
            bar = daily_data[d]
            open_p = bar.get("open")
            close_p = bar.get("close")
            volume = bar.get("volume")
            amount = bar.get("amount")
            if all(v is not None for v in (open_p, close_p, volume, amount)) and amount > 0:
                o = Decimal(str(open_p))
                c = Decimal(str(close_p))
                amt = Decimal(str(amount))
                mf = (c - o) / o * Decimal(str(volume)) / amt if o > 0 else Decimal("0")
                total_mf += mf
        return total_mf.quantize(Decimal("0.0001"))


class TurnoverAnomalyFactor(FactorBase):
    """换手率异常 = (当日换手率 - 20日均值) / 20日标准差。方向: 0。"""
    name = "turnover_anomaly"
    category = FactorCategory.SENTIMENT
    description = "换手率异常 (Z-Score)"
    direction = 0

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if not daily_data or len(daily_data) < 21:
            return None
        sorted_dates = sorted(daily_data.keys())
        recent = [Decimal(str(daily_data[d].get("turnover_rate", 0)))
                  for d in sorted_dates[-21:]]
        if len(recent) < 21:
            return None

        today = recent[-1]
        mean = sum(recent[:-1]) / (len(recent) - 1)
        variance = sum((r - mean) ** 2 for r in recent[:-1]) / (len(recent) - 1)
        std = variance.sqrt()
        if std == 0:
            return Decimal("0")
        return ((today - mean) / std).quantize(Decimal("0.0001"))


class NorthBoundFactor(FactorBase):
    """北向资金因子: 沪深港通净买入变化。方向: 1。"""
    name = "north_bound"
    category = FactorCategory.SENTIMENT
    description = "北向资金净流入变化"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        # 北向资金数据需从 shuju 层获取，这里返回 None 表示不可计算
        north_bound_data = kwargs.get("north_bound_data", {})
        if code in north_bound_data:
            return Decimal(str(north_bound_data[code]))
        return None
