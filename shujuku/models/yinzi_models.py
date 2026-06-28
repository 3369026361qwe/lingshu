"""
因子与权重 ORM 模型。

包含:
    FactorValue    — 因子值 (日频)
    FactorWeight   — 卡尔曼滤波时变权重
    FactorICRecord — 因子 IC 记录
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from shujuku.models import Base, utcnow


class FactorValue(Base):
    """单只股票的单日因子值。"""

    __tablename__ = "factor_value"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, comment="交易日")
    category: Mapped[str] = mapped_column(String(20), nullable=False, comment="因子类别 value/momentum/quality/volatility/sentiment/alternative/ai")
    factor_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="因子名 pe/pb/roe/momentum_1m ...")
    raw_value: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False, comment="原始值")
    z_score: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True, comment="标准化后 Z-Score")
    percentile: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True, comment="全市场分位数 [0,1]")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("code", "trade_date", "factor_name", name="uq_factor_value_code_date_name"),
        Index("ix_factor_value_code", "code"),
        Index("ix_factor_value_date_name", "trade_date", "factor_name"),
        Index("ix_factor_value_category", "category"),
        {"comment": "因子值表"},
    )

    def __repr__(self) -> str:
        return f"<FactorValue code={self.code} date={self.trade_date} name={self.factor_name}>"


class FactorWeight(Base):
    """卡尔曼滤波估计的时变因子权重。"""

    __tablename__ = "factor_weight"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, comment="交易日")
    category: Mapped[str] = mapped_column(String(20), nullable=False, default="", comment="因子类别")
    factor_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="因子名")
    weight: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False, comment="卡尔曼滤波权重")
    variance: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True, comment="估计方差")
    is_significant: Mapped[bool] = mapped_column(default=True, comment="是否统计显著")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("trade_date", "factor_name", name="uq_factor_weight_date_name"),
        Index("ix_factor_weight_date", "trade_date"),
        {"comment": "因子时变权重表"},
    )

    def __repr__(self) -> str:
        return f"<FactorWeight date={self.trade_date} name={self.factor_name} w={self.weight}>"


class FactorICRecord(Base):
    """因子信息系数 (IC) 记录。"""

    __tablename__ = "factor_ic_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, comment="交易日")
    factor_name: Mapped[str] = mapped_column(String(50), nullable=False, comment="因子名")
    ic: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False, comment="Rank IC")
    ir: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True, comment="IR (IC均值/IC标准差)")
    ic_window: Mapped[int] = mapped_column(Integer, default=20, comment="IC计算窗口 (交易日数)")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("trade_date", "factor_name", name="uq_ic_record_date_name"),
        Index("ix_ic_record_date", "trade_date"),
        {"comment": "因子IC记录表"},
    )

    def __repr__(self) -> str:
        return f"<FactorICRecord date={self.trade_date} name={self.factor_name} ic={self.ic}>"
