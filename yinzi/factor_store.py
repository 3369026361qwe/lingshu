"""
因子持久化存储。

将计算完成的因子值和卡尔曼权重写入 shujuku 数据库。

Usage:
    store = FactorStore(repository)
    store.save_factor_values(date, factor_results)
    store.save_factor_weights(date, factor_name, weight, variance)
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

from shujuku.repository import Repository
from yinzi.factor_base import FactorResult

_logger = logging.getLogger(__name__)


class FactorStore:
    """因子持久化存储管理器。"""

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    # ── 因子值 ────────────────────────────────────────

    def save_factor_values(
        self,
        trade_date: date,
        results: list[FactorResult],
    ) -> int:
        """批量保存因子计算结果。

        Args:
            trade_date: 交易日
            results: 因子计算结果列表

        Returns:
            成功保存的记录数
        """
        saved = 0
        for r in results:
            try:
                self._repo.save_factor_value(
                    code=r.code,
                    trade_date=trade_date,
                    category=r.category.value,
                    factor_name=r.factor_name,
                    raw_value=r.raw_value,
                    z_score=r.z_score,
                    percentile=r.percentile,
                )
                saved += 1
            except Exception as exc:
                _logger.warning(
                    "Failed to save factor %s for %s: %s",
                    r.factor_name, r.code, exc,
                )
        return saved

    def save_factor_batch(
        self,
        trade_date: date,
        code: str,
        factor_map: dict[str, Decimal],  # {factor_name: raw_value}
        category: str = "",
        z_score_map: Optional[dict[str, Decimal]] = None,
        percentile_map: Optional[dict[str, Decimal]] = None,
    ) -> int:
        """保存单只股票的全部因子值。

        Args:
            trade_date: 交易日
            code: 股票代码
            factor_map: {factor_name: raw_value}
            category: 因子类别（空字符串表示从 factor_name 推导）
            z_score_map: {factor_name: z_score}
            percentile_map: {factor_name: percentile}

        Returns:
            成功保存的记录数
        """
        saved = 0
        for factor_name, raw_value in factor_map.items():
            try:
                cat = category or self._infer_category(factor_name)
                self._repo.save_factor_value(
                    code=code,
                    trade_date=trade_date,
                    category=cat,
                    factor_name=factor_name,
                    raw_value=raw_value,
                    z_score=z_score_map.get(factor_name) if z_score_map else None,
                    percentile=percentile_map.get(factor_name) if percentile_map else None,
                )
                saved += 1
            except Exception as exc:
                _logger.warning("Failed to save factor %s: %s", factor_name, exc)
        return saved

    # ── 因子权重 ───────────────────────────────────────

    def save_factor_weights(
        self,
        trade_date: date,
        factor_names: list[str],
        weights: list[Decimal],
        variances: Optional[list[Decimal]] = None,
    ) -> int:
        """批量保存卡尔曼滤波因子权重。

        Returns:
            成功保存的记录数
        """
        saved = 0
        for i, name in enumerate(factor_names):
            var = variances[i] if variances and i < len(variances) else None
            try:
                self._repo.save_factor_weight(
                    trade_date=trade_date,
                    factor_name=name,
                    weight=weights[i],
                    variance=var,
                )
                saved += 1
            except Exception as exc:
                _logger.warning("Failed to save weight for %s: %s", name, exc)
        return saved

    def save_ic_record(
        self,
        trade_date: date,
        factor_name: str,
        ic: Decimal,
        ir: Optional[Decimal] = None,
        ic_window: int = 20,
    ) -> None:
        """保存因子 IC 记录。"""
        from shujuku.models.yinzi_models import FactorICRecord
        record = FactorICRecord(
            trade_date=trade_date,
            factor_name=factor_name,
            ic=ic,
            ir=ir,
            ic_window=ic_window,
        )
        self._repo.add(record)

    # ── 读取 ──────────────────────────────────────────

    def get_factor_values(
        self, code: str, factor_name: str, start: date, end: date
    ) -> list:
        """读取历史因子值。"""
        return self._repo.get_factor_values(code, factor_name, start, end)

    # ── 工具 ──────────────────────────────────────────

    @staticmethod
    def _infer_category(factor_name: str) -> str:
        """从因子名推断类别。"""
        value_factors = {"pe", "pb", "ps", "fcf_yield", "peg"}
        momentum_factors = {"momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m1m", "turnover_momentum"}
        quality_factors = {"roe", "roa", "gross_margin", "net_margin", "cashflow_to_revenue"}
        volatility_factors = {"historical_vol", "downside_vol", "beta", "var_95"}
        sentiment_factors = {"volume_ratio", "money_flow", "turnover_anomaly", "north_bound"}
        alternative_factors = {"analyst_coverage", "institutional_holding", "shareholder_count"}

        if factor_name in value_factors:
            return "value"
        if factor_name in momentum_factors:
            return "momentum"
        if factor_name in quality_factors:
            return "quality"
        if factor_name in volatility_factors:
            return "volatility"
        if factor_name in sentiment_factors:
            return "sentiment"
        if factor_name in alternative_factors:
            return "alternative"
        return ""
