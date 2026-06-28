"""
yinzi — 量化因子引擎

25+ 因子计算 + 卡尔曼滤波动态权重 + 因子有效性检验 + 持久化。

因子分类:
    估值   — PE, PB, PS, FCF Yield, PEG
    动量   — 1M, 3M, 6M, 12-1M, 换手率动量
    质量   — ROE, ROA, 毛利率, 净利率, 现金流/营收
    波动率 — 历史波动率, 下行波动率, Beta, VaR
    情绪   — 成交量比, 资金流向, 换手率异常, 北向资金
    另类   — 分析师覆盖变化, 机构持股变化, 股东人数变化

Usage:
    from yinzi import FactorEngine
    engine = FactorEngine()
    factors = engine.compute_all(daily_data, financial_data)
"""

from yinzi.alternative_factors import (
    AnalystCoverageFactor,
    InstitutionalHoldingFactor,
    ShareholderCountFactor,
)
from yinzi.engine import FactorEngine  # noqa: F401 (re-exported for public API)
from yinzi.factor_base import FactorBase, FactorCategory
from yinzi.factor_store import FactorStore
from yinzi.factor_validator import FactorValidator
from yinzi.kalman_weight import KalmanWeightEstimator
from yinzi.momentum_factors import (
    Momentum1MFactor,
    Momentum3MFactor,
    Momentum6MFactor,
    Momentum12M1MFactor,
)
from yinzi.quality_factors import (
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
from yinzi.value_factors import (
    FCFYieldFactor,
    PBFactor,
    PEFactor,
    PEGFactor,
    PSFactor,
)
from yinzi.volatility_factors import (
    BetaFactor,
    DownsideVolFactor,
    HistoricalVolFactor,
    VaRFactor,
)

__all__ = [
    "FactorBase", "FactorCategory",
    "PEFactor", "PBFactor", "PSFactor", "FCFYieldFactor", "PEGFactor",
    "Momentum1MFactor", "Momentum3MFactor", "Momentum6MFactor", "Momentum12M1MFactor",
    "ROEFactor", "ROAFactor", "GrossMarginFactor", "NetMarginFactor",
    "HistoricalVolFactor", "DownsideVolFactor", "BetaFactor", "VaRFactor",
    "VolumeRatioFactor", "MoneyFlowFactor", "TurnoverAnomalyFactor", "NorthBoundFactor",
    "AnalystCoverageFactor", "InstitutionalHoldingFactor", "ShareholderCountFactor",
    "KalmanWeightEstimator", "FactorValidator", "FactorStore",
]
