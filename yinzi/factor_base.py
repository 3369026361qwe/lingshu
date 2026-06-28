"""
因子抽象基类。

所有因子必须继承 FactorBase 并实现 compute() 方法。
提供统一的因子元数据、类别枚举和结果格式。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

try:
    from typing import TypedDict  # Python 3.8+
except ImportError:
    TypedDict = dict  # type: ignore

from yinzi.metrics import factor_compute_duration, factor_compute_total


class FinancialDataDict(TypedDict, total=False):
    """财务数据结构定义（IDE 提示用）。"""
    pe: float
    pb: float
    ps: float
    roe: float
    roa: float
    gross_margin: float
    net_margin: float
    revenue: float
    net_profit: float
    operating_cashflow: float
    free_cashflow_yield: float
    shareholder_count: int
    net_profit_growth: float


class DailyDataDict(TypedDict, total=False):
    """日线数据结构定义（IDE 提示用）。"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    turnover_rate: float


class FactorCategory(str, Enum):
    """因子类别枚举。"""
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    ALTERNATIVE = "alternative"
    AI = "ai"


@dataclass
class FactorResult:
    """单只股票的单因子计算结果。"""
    code: str
    factor_name: str
    category: FactorCategory
    raw_value: Decimal
    z_score: Decimal | None = None
    percentile: Decimal | None = None


class FactorBase(ABC):
    """因子抽象基类。

    子类必须:
        - 设置 name, category, description
        - 实现 compute(code, daily_data, financial_data, **kwargs) 方法
    """

    name: str = ""
    category: FactorCategory = FactorCategory.VALUE
    description: str = ""
    # 因子方向: 1 = 越大越好, -1 = 越小越好, 0 = 中性
    direction: int = 1

    @abstractmethod
    def compute(
        self,
        code: str,
        daily_data: dict,       # {trade_date: {open, high, low, close, volume, ...}}
        financial_data: dict | None = None,  # {report_date: {pe, roe, ...}}
        **kwargs,
    ) -> Decimal | None:
        """计算单只股票的最新因子值。

        Args:
            code: 6 位股票代码
            daily_data: 日线数据 (最近 N 个交易日)
            financial_data: 财务数据 (最近报告期)
            **kwargs: 额外上下文 (行业数据、市场数据等)

        Returns:
            因子原始值，无法计算时返回 None
        """
        ...

    def supports_vectorized(self) -> bool:
        """子类覆盖返回 True 表示支持向量化批量计算。"""
        return False

    def compute_vectorized(
        self,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
        **kwargs,
    ) -> list[FactorResult]:
        """向量化批量计算（子类覆盖实现）。"""
        return self.compute_batch(stock_list, daily_data_map, financial_data_map, **kwargs)

    def compute_batch(
        self,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
        **kwargs,
    ) -> list[FactorResult]:
        """批量计算因子值（默认逐只调用 compute），含 Prometheus 指标。"""
        results = []
        for code in stock_list:
            t0 = time.perf_counter()
            daily = daily_data_map.get(code, {})
            fin = (financial_data_map or {}).get(code)
            try:
                value = self.compute(code, daily, fin, **kwargs)
                if value is not None:
                    results.append(FactorResult(
                        code=code,
                        factor_name=self.name,
                        category=self.category,
                        raw_value=value,
                    ))
                    factor_compute_total.labels(factor_name=self.name, status="success").inc()
                else:
                    factor_compute_total.labels(factor_name=self.name, status="skipped").inc()
            except Exception:
                factor_compute_total.labels(factor_name=self.name, status="failure").inc()
            factor_compute_duration.labels(factor_name=self.name).observe(time.perf_counter() - t0)
        return results

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} category={self.category.value}>"
