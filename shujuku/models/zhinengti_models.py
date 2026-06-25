"""
智能体报告 ORM 模型。

每个 Agent 的分析输出通过此模型持久化，前端可直接查询展示。
AgentOutput 通信协议映射到 AgentReport + AgentEvidence 两张表。
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shujuku.models import Base, utcnow


class AgentReport(Base):
    """智能体分析报告（对应设计文档 AgentOutput）。"""

    __tablename__ = "agent_report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(30), nullable=False, comment="智能体标识 macro/sector/stock/sentiment/risk")
    analysis_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="分析日期")
    target_stocks: Mapped[str] = mapped_column(Text, nullable=False, comment="覆盖股票列表 (JSON 数组)")
    signal: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, comment="信号 [-1, 1]")
    confidence: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, comment="置信度 [0, 1]")
    reasoning: Mapped[str] = mapped_column(Text, nullable=False, comment="推理过程 (前端展示核心)")
    risk_flags: Mapped[str | None] = mapped_column(Text, nullable=True, comment="风险点 (JSON 数组)")
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True, comment="LLM 原始响应 (调试用)")

    # 元数据
    model_used: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="使用的 LLM 模型")
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="消耗 Token 数")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="响应延迟 (毫秒)")
    is_cached: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否来自缓存")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # 关联证据
    evidence: Mapped[list["AgentEvidence"]] = relationship(
        "AgentEvidence",
        back_populates="report",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_agent_report_agent_date", "agent_id", "analysis_date"),
        Index("ix_agent_report_date", "analysis_date"),
        {"comment": "智能体分析报告表"},
    )

    def __repr__(self) -> str:
        return f"<AgentReport agent={self.agent_id} date={self.analysis_date} signal={self.signal}>"


class AgentEvidence(Base):
    """Agent 报告的支撑证据。"""

    __tablename__ = "agent_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agent_report.id", ondelete="CASCADE"),
        nullable=False,
        comment="关联的报告 ID",
    )
    source: Mapped[str] = mapped_column(String(100), nullable=False, comment="数据来源")
    metric: Mapped[str] = mapped_column(String(100), nullable=False, comment="指标名")
    value: Mapped[str] = mapped_column(String(200), nullable=False, comment="指标值")

    # 反关联
    report: Mapped["AgentReport"] = relationship("AgentReport", back_populates="evidence")

    __table_args__ = (
        Index("ix_agent_evidence_report", "report_id"),
        {"comment": "Agent 报告证据表"},
    )

    def __repr__(self) -> str:
        return f"<AgentEvidence source={self.source} metric={self.metric}>"
