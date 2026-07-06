"""
jingsuan — 精算中台 (v4.1)

纯计算引擎层：无状态、无 IO、无数据库读写。
每个模块接收数学输入、返回数学结果，可独立验证。

v4.1 增强 (feat/jingsuan-enhance):
    evt_engine      — MLE GPD (Newton-Raphson) + Profile Likelihood CI + Auto Threshold + GEV
    gev_engine       — [NEW] GEV Block Maxima 分布拟合 + Return Level
    copula_engine   — Rotated Gumbel (180°) + 多维 Copula + Cramér-von Mises GoF
    dcc_copula       — [NEW] DCC-GARCH 时变条件相关 Copula
    ruin_engine     — Cramér-Lundberg 精确解 + Beekman-Bowers 近似 + 多期条件更新
    credibility     — 分层信度 + Hachemeister 回归信度 + 时序衰减加权
    reserving       — [NEW] Chain Ladder / Bornhuetter-Ferguson 准备金方法
    var_backtest    — Dynamic Quantile (DQ) 检验 + Berkowitz LR + 滚动 Traffic Light
    solvency        — Solvency II SCR 风险资本 (unchanged)
    scenario_gen    — 随机情景生成器 (unchanged)
    stress_engine   — 反向压力测试 (unchanged)
    risk_budget     — 动态风险预算引擎 (unchanged)
"""

__version__ = "4.1.0"

# ruff: noqa: F401 — re-exports for public API
from jingsuan.copula_engine import (
    CopulaEngine,
    CopulaFit,
    CopulaType,
    MultiCopulaFit,
)
from jingsuan.credibility import (
    CredibilityEngine,
    CredibilityWeights,
    HachemeisterResult,
    HierarchicalCredibilityResult,
    SourceTrackRecord,
    TimeDecayConfig,
)
from jingsuan.dcc_copula import (
    DCCCopula,
    DCCFitResult,
)
from jingsuan.evt_engine import (
    EVTEngine,
    EVTFitResult,
    EVTVaRResult,
    ProfileLikelihoodCI,
)
from jingsuan.gev_engine import (
    GEVEngine,
    GEVFitResult,
    GEVReturnLevel,
)
from jingsuan.reserving import (
    LossTriangle,
    ReservingEngine,
    ReservingResult,
)
from jingsuan.risk_budget import RiskBudgetEngine, RiskLimits
from jingsuan.ruin_engine import (
    MultiPeriodRuin,
    RuinAnalysis,
    RuinConfig,
    RuinEngine,
)
from jingsuan.scenario_gen import ScenarioGenerator
from jingsuan.solvency import (
    SCRCalculator,
    SCRDecomposition,
)
from jingsuan.stress_engine import StressEngine, StressScenario
from jingsuan.var_backtest import (
    RollingBacktestResult,
    VaRBacktestResult,
    VaRBacktestSuite,
)
