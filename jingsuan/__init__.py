"""
jingsuan — 精算中台 (v4.0)

纯计算引擎层：无状态、无 IO、无数据库读写。
每个模块接收数学输入、返回数学结果，可独立验证。

Modules:
    evt_engine      — 极值理论 GPD/POT 尾部 VaR
    copula_engine   — Copula 连接函数多资产尾部依赖
    ruin_engine     — 破产理论动态风险预算
    credibility     — 信度理论信号融合
    solvency        — Solvency II SCR 风险资本
    scenario_gen    — 随机情景生成器
    var_backtest    — VaR 回测检验套件
    stress_engine   — 反向压力测试
    risk_budget     — 动态风险预算引擎
"""

from .copula_engine import CopulaEngine, CopulaType
from .credibility import CredibilityEngine, SourceTrackRecord
from .evt_engine import EVTEngine
from .risk_budget import RiskBudgetEngine
from .ruin_engine import RuinConfig, RuinEngine
from .scenario_gen import ScenarioGenerator
from .solvency import SCRCalculator
from .stress_engine import StressEngine
from .var_backtest import VaRBacktestSuite

__all__ = [
    "CopulaEngine", "CopulaType",
    "CredibilityEngine", "SourceTrackRecord",
    "EVTEngine",
    "RiskBudgetEngine",
    "RuinEngine", "RuinConfig",
    "ScenarioGenerator",
    "SCRCalculator",
    "StressEngine",
    "VaRBacktestSuite",
]

__version__ = "4.0.0"
