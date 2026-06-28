"""
风控记录 ORM 模型。

包含:
    CircuitBreakerEvent — 熔断器状态变更事件
    RiskLog             — 风控日志
    VaRRecord           — VaR/CVaR 每日记录
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from shujuku.models import Base, utcnow


class CircuitBreakerEvent(Base):
    """熔断器三态切换事件日志。"""

    __tablename__ = "circuit_breaker_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="事件时间")
    from_state: Mapped[str] = mapped_column(String(20), nullable=False, comment="切换前状态 CLOSED/OPEN/HALF_OPEN")
    to_state: Mapped[str] = mapped_column(String(20), nullable=False, comment="切换后状态 CLOSED/OPEN/HALF_OPEN")
    trigger_reason: Mapped[str] = mapped_column(String(200), nullable=False, comment="触发原因")
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True, comment="触发时关键指标 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_cb_events_timestamp", "timestamp"),
        Index("ix_cb_events_state", "to_state"),
        {"comment": "熔断器事件表"},
    )

    def __repr__(self) -> str:
        return f"<CircuitBreakerEvent {self.from_state}→{self.to_state} reason={self.trigger_reason}>"


class RiskLog(Base):
    """风控日志 (通用)。"""

    __tablename__ = "risk_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="日志时间")
    level: Mapped[str] = mapped_column(String(10), nullable=False, comment="INFO/WARNING/CRITICAL")
    category: Mapped[str] = mapped_column(String(30), nullable=False, comment="风控类别 position/var/concentration/circuit_breaker/black_swan")
    message: Mapped[str] = mapped_column(String(500), nullable=False, comment="日志消息")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True, comment="详细数据 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_risk_logs_timestamp", "timestamp"),
        Index("ix_risk_logs_level", "level"),
        Index("ix_risk_logs_category", "category"),
        {"comment": "风控日志表"},
    )

    def __repr__(self) -> str:
        return f"<RiskLog level={self.level} category={self.category}>"


class VaRRecord(Base):
    """VaR/CVaR 每日记录。"""

    __tablename__ = "var_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calc_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="计算日期")
    confidence_level: Mapped[Decimal] = mapped_column(Numeric(4, 2), nullable=False, comment="置信水平 0.95/0.99")
    var: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="VaR 值")
    cvar: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="CVaR 值")
    method: Mapped[str] = mapped_column(String(20), default="historical", comment="计算方法 historical/parametric/monte_carlo")
    window_days: Mapped[int] = mapped_column(Integer, default=252, comment="计算窗口 (交易日)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_var_records_date", "calc_date"),
        {"comment": "VaR记录表"},
    )

    def __repr__(self) -> str:
        return f"<VaRRecord date={self.calc_date} var={self.var} cvar={self.cvar}>"
