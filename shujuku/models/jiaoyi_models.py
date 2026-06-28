"""
交易与持仓 ORM 模型。

包含:
    Order              — 订单 (指令)
    Trade              — 成交记录
    Position           — 当前持仓
    PortfolioSnapshot  — 组合快照 (日频)
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from shujuku.models import Base, utcnow


class Order(Base):
    """交易订单。"""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="订单ID (UUID)")
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    direction: Mapped[str] = mapped_column(String(10), nullable=False, comment="BUY / SELL")
    order_type: Mapped[str] = mapped_column(String(20), default="LIMIT", comment="LIMIT / MARKET")
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, comment="委托数量 (股)")
    price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="委托价格")
    status: Mapped[str] = mapped_column(String(20), default="PENDING", comment="PENDING/FILLED/PARTIAL/CANCELLED/REJECTED")
    filled_qty: Mapped[int] = mapped_column(Integer, default=0, comment="已成交数量")
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="成交均价")
    reason: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="下单原因 (信号来源)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_orders_code", "code"),
        Index("ix_orders_status", "status"),
        Index("ix_orders_created", "created_at"),
        {"comment": "交易订单表"},
    )

    def __repr__(self) -> str:
        return f"<Order id={self.order_id} code={self.code} dir={self.direction} status={self.status}>"


class Trade(Base):
    """成交记录。"""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="成交ID")
    order_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="关联订单ID")
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    direction: Mapped[str] = mapped_column(String(10), nullable=False, comment="BUY / SELL")
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, comment="成交数量 (股)")
    price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="成交价格")
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="成交金额")
    commission: Mapped[Decimal] = mapped_column(Numeric(18, 4), default=0, comment="手续费")
    trade_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="成交时间")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_trades_code", "code"),
        Index("ix_trades_order", "order_id"),
        Index("ix_trades_time", "trade_time"),
        {"comment": "成交记录表"},
    )

    def __repr__(self) -> str:
        return f"<Trade id={self.trade_id} code={self.code} qty={self.quantity} price={self.price}>"


class Position(Base):
    """实时持仓。"""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, comment="股票代码")
    name: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="股票名称")
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="持仓数量")
    avg_cost: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, comment="持仓均价")
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True, comment="最新市价")
    market_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="市值")
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(Numeric(18, 2), nullable=True, comment="未实现盈亏")
    weight: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True, comment="占总仓位权重")
    industry: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="申万一级行业")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("code", name="uq_positions_code"),
        {"comment": "实时持仓表"},
    )

    def __repr__(self) -> str:
        return f"<Position code={self.code} qty={self.quantity} cost={self.avg_cost}>"


class PortfolioSnapshot(Base):
    """组合快照 (每日收盘后记录)。"""

    __tablename__ = "portfolio_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, comment="交易日")
    total_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="总资产")
    cash: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="现金")
    market_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, comment="持仓市值")
    daily_return: Mapped[Decimal | None] = mapped_column(Numeric(12, 8), nullable=True, comment="日收益率")
    cumulative_return: Mapped[Decimal | None] = mapped_column(Numeric(12, 8), nullable=True, comment="累计收益率")
    position_count: Mapped[int] = mapped_column(Integer, default=0, comment="持仓股票数")
    leverage: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True, comment="杠杆率")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("trade_date", name="uq_portfolio_snapshot_date"),
        Index("ix_portfolio_snapshot_date", "trade_date"),
        {"comment": "组合快照表"},
    )

    def __repr__(self) -> str:
        return f"<PortfolioSnapshot date={self.trade_date} value={self.total_value}>"
