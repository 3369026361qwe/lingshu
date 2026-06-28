"""
因子计算统一引擎。

提供并行/串行双模式，支持向量化和传统因子混合调度。

Usage:
    from yinzi import create_default_engine
    engine = create_default_engine()
    results = engine.compute_all(stocks, daily_map, fin_map)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from yinzi.factor_base import FactorBase, FactorResult
from yinzi.metrics import factor_batch_duration

_logger = logging.getLogger(__name__)


class FactorEngine:
    """因子计算统一引擎。

    管理因子注册、批量计算（串行/并行）、按类别分组。
    """

    def __init__(self, max_workers: int = 8):
        self._factors: list[FactorBase] = []
        self._max_workers = max_workers

    def register(self, factor: FactorBase) -> None:
        """注册因子。"""
        self._factors.append(factor)

    def register_all(self, factors: list[FactorBase]) -> None:
        """批量注册因子。"""
        self._factors.extend(factors)

    @property
    def factor_count(self) -> int:
        return len(self._factors)

    def get_factors_by_category(self, category: str) -> list[FactorBase]:
        """按类别筛选因子。"""
        from yinzi.factor_base import FactorCategory
        cat = FactorCategory(category)
        return [f for f in self._factors if f.category == cat]

    # ── 计算入口 ────────────────────────────────────────

    def compute_all(
        self,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
        parallel: bool = True,
    ) -> list[FactorResult]:
        """计算全部已注册因子。

        Args:
            stock_list: 股票代码列表
            daily_data_map: {code: {trade_date: bar}}
            financial_data_map: {code: {report_date: fin_data}}
            parallel: 因子级别并行

        Returns:
            全部因子计算结果列表
        """
        t0 = time.perf_counter()

        if parallel and len(self._factors) > 1:
            results = self._compute_parallel(stock_list, daily_data_map, financial_data_map)
        else:
            results = self._compute_sequential(stock_list, daily_data_map, financial_data_map)

        factor_batch_duration.labels(category="all").observe(time.perf_counter() - t0)
        return results

    def compute_category(
        self,
        category: str,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
    ) -> list[FactorResult]:
        """计算指定类别的因子。"""
        factors = self.get_factors_by_category(category)
        t0 = time.perf_counter()
        results = []
        for f in factors:
            batch_results = self._compute_one_factor(f, stock_list, daily_data_map, financial_data_map)
            results.extend(batch_results)
        factor_batch_duration.labels(category=category).observe(time.perf_counter() - t0)
        return results

    # ── 内部 ────────────────────────────────────────────

    def _compute_one_factor(
        self,
        factor: FactorBase,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
    ) -> list[FactorResult]:
        """计算单个因子。根据因子能力选择向量化或普通路径。"""
        if factor.supports_vectorized():
            return factor.compute_vectorized(stock_list, daily_data_map, financial_data_map)
        return factor.compute_batch(stock_list, daily_data_map, financial_data_map)

    def _compute_sequential(
        self,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
    ) -> list[FactorResult]:
        """串行计算所有因子。"""
        results = []
        for f in self._factors:
            results.extend(self._compute_one_factor(f, stock_list, daily_data_map, financial_data_map))
        return results

    def _compute_parallel(
        self,
        stock_list: list[str],
        daily_data_map: dict[str, dict],
        financial_data_map: dict[str, dict] | None = None,
    ) -> list[FactorResult]:
        """因子级别并行计算。"""
        results = []
        n_workers = min(self._max_workers, len(self._factors))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for f in self._factors:
                future = executor.submit(
                    self._compute_one_factor, f, stock_list, daily_data_map, financial_data_map
                )
                futures[future] = f

            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    _logger.error("Factor %s compute failed: %s", futures[future].name, exc, exc_info=True)
        return results


# ── 工厂函数 ──────────────────────────────────────────

def create_default_engine(max_workers: int = 8) -> FactorEngine:
    """创建包含全部 21 个因子的预配置引擎。"""
    from yinzi.alternative_factors import (
        AnalystCoverageFactor,
        InstitutionalHoldingFactor,
        ShareholderCountFactor,
    )
    from yinzi.momentum_factors import (
        Momentum1MFactor,
        Momentum3MFactor,
        Momentum6MFactor,
        Momentum12M1MFactor,
        TurnoverMomentumFactor,
    )
    from yinzi.quality_factors import (
        CashflowToRevenueFactor,
        GrossMarginFactor,
        NetMarginFactor,
        ROAFactor,
        ROEFactor,
    )
    from yinzi.sentiment_factors import (
        MoneyFlowFactor,
        NorthBoundFactor,
        TurnoverAnomalyFactor,
        VolumeRatioFactor,
    )
    from yinzi.value_factors import FCFYieldFactor, PBFactor, PEFactor, PEGFactor, PSFactor
    from yinzi.volatility_factors import (
        BetaFactor,
        DownsideVolFactor,
        HistoricalVolFactor,
        VaRFactor,
    )

    engine = FactorEngine(max_workers=max_workers)

    engine.register_all([
        PEFactor(), PBFactor(), PSFactor(), FCFYieldFactor(), PEGFactor(),
        Momentum1MFactor(), Momentum3MFactor(), Momentum6MFactor(),
        Momentum12M1MFactor(), TurnoverMomentumFactor(),
        ROEFactor(), ROAFactor(), GrossMarginFactor(), NetMarginFactor(), CashflowToRevenueFactor(),
        HistoricalVolFactor(), DownsideVolFactor(), BetaFactor(), VaRFactor(),
        VolumeRatioFactor(), MoneyFlowFactor(), TurnoverAnomalyFactor(), NorthBoundFactor(),
        AnalystCoverageFactor(), InstitutionalHoldingFactor(), ShareholderCountFactor(),
    ])

    return engine
