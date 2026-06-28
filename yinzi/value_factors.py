"""
估值因子: PE, PB, PS, FCF Yield, PEG。

数据来源: Tushare 财务数据 (financial_data)。
"""

from decimal import Decimal

from yinzi.factor_base import FactorBase, FactorCategory, FactorResult


class PEFactor(FactorBase):
    """市盈率 = 股价 / 每股收益。方向: -1 (越低越好)。"""

    name = "pe"
    category = FactorCategory.VALUE
    description = "市盈率 (Price to Earnings)"
    direction = -1

    def supports_vectorized(self) -> bool: return True

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if financial_data:
            pe = financial_data.get("pe")
            if pe is not None:
                return Decimal(str(pe))
        return None

    def compute_vectorized(self, stock_list, daily_data_map, financial_data_map=None, **kwargs):
        results = []
        for code in stock_list:
            fin = (financial_data_map or {}).get(code, {})
            pe = fin.get("pe")
            if pe is not None:
                results.append(FactorResult(code=code, factor_name=self.name, category=self.category, raw_value=Decimal(str(pe))))
        return results


class PBFactor(FactorBase):
    """市净率 = 股价 / 每股净资产。方向: -1。"""

    name = "pb"
    category = FactorCategory.VALUE
    description = "市净率 (Price to Book)"
    direction = -1

    def supports_vectorized(self) -> bool: return True

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if financial_data:
            pb = financial_data.get("pb")
            if pb is not None:
                return Decimal(str(pb))
        return None

    def compute_vectorized(self, stock_list, daily_data_map, financial_data_map=None, **kwargs):
        results = []
        for code in stock_list:
            fin = (financial_data_map or {}).get(code, {})
            pb = fin.get("pb")
            if pb is not None:
                results.append(FactorResult(code=code, factor_name=self.name, category=self.category, raw_value=Decimal(str(pb))))
        return results


class PSFactor(FactorBase):
    """市销率 = 总市值 / 营业收入。方向: -1。"""

    name = "ps"
    category = FactorCategory.VALUE
    description = "市销率 (Price to Sales)"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if financial_data:
            ps = financial_data.get("ps")
            if ps is not None:
                return Decimal(str(ps))
        price = self._latest_close(daily_data)
        revenue = financial_data.get("revenue") if financial_data else None
        if price and revenue and revenue > 0:
            return price / Decimal(str(revenue))
        return None

    @staticmethod
    def _latest_close(daily_data: dict) -> Decimal | None:
        if not daily_data:
            return None
        latest_date = max(daily_data.keys())
        close = daily_data[latest_date].get("close")
        return Decimal(str(close)) if close else None


class FCFYieldFactor(FactorBase):
    """自由现金流收益率 = FCF / 总市值。方向: 1 (越高越好)。"""

    name = "fcf_yield"
    category = FactorCategory.VALUE
    description = "自由现金流收益率"
    direction = 1

    def supports_vectorized(self) -> bool: return True

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if financial_data:
            fcf_yield = financial_data.get("free_cashflow_yield")
            if fcf_yield is not None:
                return Decimal(str(fcf_yield))
        return None

    def compute_vectorized(self, stock_list, daily_data_map, financial_data_map=None, **kwargs):
        results = []
        for code in stock_list:
            fin = (financial_data_map or {}).get(code, {})
            fy = fin.get("free_cashflow_yield")
            if fy is not None:
                results.append(FactorResult(code=code, factor_name=self.name, category=self.category, raw_value=Decimal(str(fy))))
        return results


class PEGFactor(FactorBase):
    """PEG = PE / 盈利增长率。方向: -1。"""

    name = "peg"
    category = FactorCategory.VALUE
    description = "市盈率相对盈利增长比率"
    direction = -1

    def compute(self, code, daily_data, financial_data=None, **kwargs) -> Decimal | None:
        if not financial_data:
            return None
        pe = financial_data.get("pe")
        if pe is None:
            return None
        # P1-3: 优先外部增长，否则从历史财报自计算
        net_profit_growth = financial_data.get("net_profit_growth")
        if net_profit_growth is None:
            historical = kwargs.get("historical_financials", {})
            if code in historical:
                net_profit_growth = self._calc_growth(historical[code])
        if net_profit_growth is None or net_profit_growth <= 0:
            return None
        pe_d = Decimal(str(pe))
        growth_d = Decimal(str(net_profit_growth))
        return (pe_d / growth_d).quantize(Decimal("0.01"))

    @staticmethod
    def _calc_growth(historical_reports: list[dict]) -> float | None:
        """从历史报告计算净利润同比增长率。"""
        if len(historical_reports) < 2:
            return None
        latest = historical_reports[-1].get("net_profit")
        previous = historical_reports[-2].get("net_profit")
        if latest is None or previous is None or previous == 0:
            return None
        return float((Decimal(str(latest)) - Decimal(str(previous))) / Decimal(str(previous)))
