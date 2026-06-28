"""
SQLAlchemy ORM 模型集合。

所有模型继承自 Base。按业务域分为 5 个子模块:
    market_models    — 股票基础信息、日线行情、财务报表
    yinzi_models     — 因子值、卡尔曼权重
    zhinengti_models — 智能体输出报告
    jiaoyi_models    — 订单、成交、持仓
    fengkong_models  — 熔断事件、风控日志
"""

from datetime import datetime, timezone  # noqa: F401 (timezone used in utcnow default; re-exported)

from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    """返回 UTC 感知的当前时间，替代 Python 3.12+ 已弃用的 datetime.utcnow()。"""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """所有 ORM 模型的抽象基类。"""
    pass


# 延迟导入模型子模块以确保 Base 已定义
from shujuku.models.fengkong_models import (  # noqa: E402, F401
    CircuitBreakerEvent,
    RiskLog,
    VaRRecord,
)
from shujuku.models.jiaoyi_models import (  # noqa: E402, F401
    Order,
    PortfolioSnapshot,
    Position,
    Trade,
)
from shujuku.models.juece_models import (  # noqa: E402, F401
    FusionScore,
)
from shujuku.models.market_models import (  # noqa: E402, F401
    DailyBar,
    FinancialReport,
    IndustryClassification,
    StockInfo,
)
from shujuku.models.yinzi_models import (  # noqa: E402, F401
    FactorICRecord,
    FactorValue,
    FactorWeight,
)
from shujuku.models.zhinengti_models import (  # noqa: E402, F401
    AgentEvidence,
    AgentReport,
)

__all__ = [
    "Base",
    # market
    "StockInfo",
    "DailyBar",
    "FinancialReport",
    "IndustryClassification",
    # yinzi
    "FactorValue",
    "FactorWeight",
    "FactorICRecord",
    # zhinengti
    "AgentReport",
    "AgentEvidence",
    # jiaoyi
    "Order",
    "Trade",
    "Position",
    "PortfolioSnapshot",
    # fengkong
    "CircuitBreakerEvent",
    "RiskLog",
    "VaRRecord",
    # juece
    "FusionScore",
]
