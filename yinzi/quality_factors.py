"""
质量因子: ROE, ROA, 毛利率, 净利率, 现金流/营收。

数据来源: Tushare 财务数据。
"""

from decimal import Decimal
from typing import Optional

from yinzi.factor_base import FactorBase, FactorCategory


class ROEFactor(FactorBase):
    """ROE = 归母净利润 / 净资产。方向: 1。"""
    name = "roe"
    category = FactorCategory.QUALITY
    description = "净资产收益率"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if financial_data and financial_data.get("roe") is not None:
            return Decimal(str(financial_data["roe"]))
        return None


class ROAFactor(FactorBase):
    """ROA = 净利润 / 总资产。方向: 1。"""
    name = "roa"
    category = FactorCategory.QUALITY
    description = "总资产收益率"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if financial_data and financial_data.get("roa") is not None:
            return Decimal(str(financial_data["roa"]))
        return None


class GrossMarginFactor(FactorBase):
    """毛利率 = (营收 - 营业成本) / 营收。方向: 1。"""
    name = "gross_margin"
    category = FactorCategory.QUALITY
    description = "毛利率"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if financial_data and financial_data.get("gross_margin") is not None:
            return Decimal(str(financial_data["gross_margin"]))
        return None


class NetMarginFactor(FactorBase):
    """净利率 = 净利润 / 营收。方向: 1。"""
    name = "net_margin"
    category = FactorCategory.QUALITY
    description = "净利率"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if financial_data and financial_data.get("net_margin") is not None:
            return Decimal(str(financial_data["net_margin"]))
        return None


class CashflowToRevenueFactor(FactorBase):
    """经营现金流 / 营收。方向: 1。"""
    name = "cashflow_to_revenue"
    category = FactorCategory.QUALITY
    description = "经营现金流/营收"
    direction = 1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Optional[Decimal]:
        if not financial_data:
            return None
        ocf = financial_data.get("operating_cashflow")
        revenue = financial_data.get("revenue")
        if ocf and revenue and revenue > 0:
            return Decimal(str(ocf)) / Decimal(str(revenue))
        return None
