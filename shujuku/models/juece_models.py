"""决策引擎相关 ORM 模型 — 融合分数、选股信号等。"""

from datetime import datetime, timezone

from sqlalchemy import Column, Date, Float, Index, Integer, String, DateTime

from shujuku.models import Base, utcnow


class FusionScore(Base):
    """因子融合综合得分。

    每行 = 某日期某股票的综合得分及排名。
    由 FactorFusion 计算后写入。
    """

    __tablename__ = "fusion_score"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False, index=True, comment="交易日期")
    code = Column(String(12), nullable=False, comment="股票代码")
    composite_score = Column(Float, nullable=False, comment="综合得分 [0,1]")
    rank = Column(Integer, nullable=False, comment="全市场排名")
    signal = Column(Float, nullable=True, comment="选股信号")
    created_at = Column(DateTime, default=utcnow, comment="创建时间")

    __table_args__ = (
        Index("ix_fusion_date_code", "trade_date", "code", unique=True),
        Index("ix_fusion_date_rank", "trade_date", "rank"),
        {"comment": "因子融合综合得分"},
    )
