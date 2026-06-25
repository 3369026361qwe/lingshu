"""
行情与财务 ORM 模型。

包含:
    StockInfo               — 股票基础信息
    DailyBar                — 日线行情 (OHLCV)
    FinancialReport         — 财务报表 (季频)
    IndustryClassification  — 行业分类 (申万)
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Float, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from shujuku.models import Base, utcnow


class StockInfo(Base):
    """A 股股票基础信息。"""

    __tablename__ = "stock_info"

    # 主键: 6 位股票代码
    code: Mapped[str] = mapped_column(String(10), primary_key=True, comment="股票代码")
    name: Mapped[str] = mapped_column(String(50), nullable=False, comment="股票名称")
    exchange: Mapped[str] = mapped_column(String(10), nullable=False, default="SZ", comment="交易所 SZ/SH/BJ")
    listing_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment="上市日期")
    delisting_date: Mapped[date | None] = mapped_column(Date, nullable=True, comment="退市日期")
    is_active: Mapped[bool] = mapped_column(default=True, comment="是否活跃")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_stock_info_exchange", "exchange"),
        Index("ix_stock_info_is_active", "is_active"),
        {"comment": "A股股票基础信息表"},
    )

    def __repr__(self) -> str:
        return f"<StockInfo code={self.code} name={self.name}>"


class DailyBar(Base):
    """日线行情数据 (OHLCV)。"""

    __tablename__ = "daily_bar"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, comment="交易日")
    open: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="开盘价")
    high: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="最高价")
    low: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="最低价")
    close: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="收盘价")
    volume: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="成交量 (股)")
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="成交额 (元)")
    turnover_rate: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True, comment="换手率")
    is_st: Mapped[bool] = mapped_column(default=False, comment="是否 ST")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("code", "trade_date", name="uq_daily_bar_code_date"),
        Index("ix_daily_bar_code", "code"),
        Index("ix_daily_bar_trade_date", "trade_date"),
        Index("ix_daily_bar_code_date", "code", "trade_date"),
        {"comment": "日线行情表"},
    )

    def __repr__(self) -> str:
        return f"<DailyBar code={self.code} date={self.trade_date} close={self.close}>"


class FinancialReport(Base):
    """财务报表数据 (季频为主，日频通过插值获得)。"""

    __tablename__ = "financial_report"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    report_date: Mapped[date] = mapped_column(Date, nullable=False, comment="报告期")
    report_type: Mapped[str] = mapped_column(String(10), nullable=False, comment="报告类型 Q1/Q2/Q3/Q4")

    # 估值指标
    pe: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="市盈率")
    pb: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="市净率")
    ps: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="市销率")
    peg: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="PEG")

    # 盈利指标
    roe: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="ROE (%)")
    roa: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="ROA (%)")
    gross_margin: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="毛利率 (%)")
    net_margin: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="净利率 (%)")

    # 现金流
    revenue: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="营业收入")
    net_profit: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="归母净利润")
    operating_cashflow: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="经营现金流")
    free_cashflow_yield: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="自由现金流收益率")

    # 股东
    shareholder_count: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="股东人数")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("code", "report_date", name="uq_fin_report_code_date"),
        Index("ix_fin_report_code", "code"),
        Index("ix_fin_report_date", "report_date"),
        {"comment": "财务报表表"},
    )

    def __repr__(self) -> str:
        return f"<FinancialReport code={self.code} date={self.report_date}>"


class IndustryClassification(Base):
    """行业分类 (申万一级/二级)。"""

    __tablename__ = "industry_classification"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    sw_level1: Mapped[str] = mapped_column(String(50), nullable=False, comment="申万一级行业")
    sw_level2: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="申万二级行业")
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, comment="生效日期")
    source: Mapped[str] = mapped_column(String(20), default="ShenWan", comment="分类来源")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_ind_class_code", "code"),
        Index("ix_ind_class_sw1", "sw_level1"),
        {"comment": "行业分类表"},
    )

    def __repr__(self) -> str:
        return f"<IndustryClassification code={self.code} sw1={self.sw_level1}>"
