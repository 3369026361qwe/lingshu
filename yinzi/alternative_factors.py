"""
另类因子: 分析师覆盖变化、机构持股变化、股东人数变化。

数据来源: Tushare 另类数据 + shuju 层预处理结果。
"""

from decimal import Decimal
from typing import Optional

from yinzi.factor_base import FactorBase, FactorCategory


class AnalystCoverageFactor(FactorBase):
    """分析师覆盖变化 = (本期覆盖数 - 上期覆盖数) / 上期覆盖数。方向: 1。"""
    name = "analyst_coverage"
    category = FactorCategory.ALTERNATIVE
    description = "分析师覆盖变化率"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        coverage_data = kwargs.get("analyst_coverage", {})
        if code in coverage_data:
            current, previous = coverage_data[code]
            if previous and previous > 0:
                return (Decimal(str(current)) - Decimal(str(previous))) / Decimal(str(previous))
        return None


class InstitutionalHoldingFactor(FactorBase):
    """机构持股变化 = (本期持股比例 - 上期持股比例)。方向: 1。"""
    name = "institutional_holding"
    category = FactorCategory.ALTERNATIVE
    description = "机构持股比例变化"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        inst_data = kwargs.get("institutional_holding", {})
        if code in inst_data:
            current, previous = inst_data[code]
            return Decimal(str(current)) - Decimal(str(previous))
        return None


class ShareholderCountFactor(FactorBase):
    """股东人数变化率 = (上期人数 - 本期人数) / 上期人数。方向: -1 (人数减少=筹码集中=利好)。"""
    name = "shareholder_count"
    category = FactorCategory.ALTERNATIVE
    description = "股东人数变化率"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        holder_data = kwargs.get("shareholder_count", {})
        if code in holder_data:
            current, previous = holder_data[code]
            if previous and previous > 0:
                # CRITICAL FIX: 标准变化率 = (current-previous)/previous
                # 股东人数下降 → 负值 → direction=-1 → 排名靠前（正确）
                return (Decimal(str(current)) - Decimal(str(previous))) / Decimal(str(previous))
        # 从 financial_data 中获取
        if financial_data and financial_data.get("shareholder_count") is not None:
            return Decimal(str(financial_data["shareholder_count"]))
        return None
