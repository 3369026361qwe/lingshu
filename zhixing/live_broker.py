"""
实盘券商抽象接口 (v4.0).

定义实盘/模拟盘统一的 AbstractBroker 接口，支持:
    - 下单 (限价/市价)
    - 撤单
    - 持仓查询
    - 账户查询
    - 订单状态追踪

Usage:
    from zhixing.live_broker import AbstractBroker, StubBroker
    broker = StubBroker()
    order_id = broker.submit_order("000001", "BUY", 1000, Decimal("10.50"))
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum


class OrderType(str, Enum):
    """订单类型."""
    MARKET = "MARKET"     # 市价单
    LIMIT = "LIMIT"       # 限价单
    STOP = "STOP"         # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单


class OrderSide(str, Enum):
    """买卖方向."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """订单状态."""
    PENDING = "PENDING"       # 待提交
    SUBMITTED = "SUBMITTED"   # 已提交
    PARTIAL = "PARTIAL"       # 部分成交
    FILLED = "FILLED"         # 全部成交
    CANCELLED = "CANCELLED"   # 已撤销
    REJECTED = "REJECTED"     # 被拒绝
    EXPIRED = "EXPIRED"       # 已过期


@dataclass
class BrokerOrder:
    """券商订单."""
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    code: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: Decimal = Decimal("0")
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_avg_price: Decimal | None = None
    commission: Decimal = Decimal("0")
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    external_id: str = ""  # 券商系统返回的订单号


@dataclass
class Position:
    """持仓."""
    code: str
    quantity: int
    avg_cost: Decimal
    market_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


@dataclass
class AccountInfo:
    """账户信息."""
    account_id: str
    total_equity: Decimal
    cash: Decimal
    market_value: Decimal
    frozen_cash: Decimal = Decimal("0")
    available_cash: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    margin_ratio: Decimal = Decimal("0")


class AbstractBroker(ABC):
    """实盘券商抽象接口.

    所有实盘券商适配器必须实现此接口。
    支持 A 股、港股、美股等不同市场的券商接入。

    方法契约:
        - submit_order: 提交订单 → 返回 order_id (券商侧)
        - cancel_order: 撤销订单 → 返回是否成功
        - get_positions: 查询当前持仓
        - get_account: 查询账户信息
        - get_order: 查询订单状态
        - get_orders: 查询所有活动订单
    """

    @abstractmethod
    def submit_order(
        self,
        code: str,
        side: str,
        quantity: int,
        price: Decimal | None = None,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> BrokerOrder:
        """提交订单.

        Args:
            code: 股票代码
            side: "BUY" | "SELL"
            quantity: 委托数量
            price: 委托价 (市价单可为 None)
            order_type: "MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT"
            reason: 下单原因 (记录用)

        Returns:
            BrokerOrder with broker-assigned order_id
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单.

        Args:
            order_id: 要撤销的订单 ID

        Returns:
            True if cancellation was successful
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """查询当前持仓.

        Returns:
            持仓列表
        """
        ...

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """查询账户信息.

        Returns:
            账户信息
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder | None:
        """查询单个订单状态.

        Args:
            order_id: 订单 ID

        Returns:
            BrokerOrder or None if not found
        """
        ...

    @abstractmethod
    def get_orders(self, status_filter: list[str] | None = None) -> list[BrokerOrder]:
        """查询活动订单.

        Args:
            status_filter: 按状态过滤 (None = 所有活动订单)

        Returns:
            订单列表
        """
        ...


class StubBroker(AbstractBroker):
    """Stub 券商实现 — 用于回测和测试.

    模拟实盘接口，所有订单立即以市价成交。
    支持基本的 A 股规则: T+1、按手取整、印花税。

    Usage:
        broker = StubBroker(initial_cash=Decimal("1000000"))
        order = broker.submit_order("000001", "BUY", 1000, Decimal("10.50"))
        positions = broker.get_positions()
    """

    A_SHARE_LOT = 100
    DEFAULT_COMMISSION_RATE = Decimal("0.0003")
    MIN_COMMISSION = Decimal("5")
    STAMP_TAX = Decimal("0.001")  # 卖出时收取

    def __init__(
        self,
        initial_cash: Decimal = Decimal("1000000"),
        commission_rate: Decimal | None = None,
        slippage: Decimal = Decimal("0.0001"),
    ):
        self._account_id = f"STUB_{uuid.uuid4().hex[:8]}"
        self._cash = initial_cash
        self._frozen_cash = Decimal("0")
        self._initial_cash = initial_cash
        self._commission_rate = commission_rate if commission_rate is not None else self.DEFAULT_COMMISSION_RATE
        self._slippage = slippage
        self._positions: dict[str, dict] = {}
        self._orders: dict[str, BrokerOrder] = {}
        self._today_buys: dict[str, int] = {}
        self._trade_history: list[dict] = []

    # ── AbstractBroker 接口实现 ──────────────────────────────

    def submit_order(
        self,
        code: str,
        side: str,
        quantity: int,
        price: Decimal | None = None,
        order_type: str = "LIMIT",
        reason: str = "",
    ) -> BrokerOrder:
        """提交订单 (Stub: 立即成交)."""
        side_enum = OrderSide(side.upper())
        qty = (quantity // self.A_SHARE_LOT) * self.A_SHARE_LOT
        if qty <= 0:
            raise ValueError(f"Quantity too small, minimum {self.A_SHARE_LOT} shares")

        order = BrokerOrder(
            code=code, side=side_enum, quantity=qty,
            price=price or Decimal("0"),
            order_type=OrderType(order_type),
            reason=reason, status=OrderStatus.SUBMITTED,
        )

        # 卖出检查
        if side_enum == OrderSide.SELL:
            pos = self._positions.get(code, {})
            available = pos.get("quantity", 0) - self._today_buys.get(code, 0)
            if available < qty:
                order.status = OrderStatus.REJECTED
                order.reason = f"持仓不足: 可用{available}股"
                self._orders[order.order_id] = order
                return order

        # 模拟成交
        fill_price = self._get_fill_price(order)
        commission = max(fill_price * Decimal(str(qty)) * self._commission_rate, self.MIN_COMMISSION)
        tax = fill_price * Decimal(str(qty)) * self.STAMP_TAX if side_enum == OrderSide.SELL else Decimal("0")

        order.filled_qty = qty
        order.filled_avg_price = fill_price
        order.commission = commission + tax
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._orders[order.order_id] = order

        # 更新持仓和资金
        trade_amount = fill_price * Decimal(str(qty))
        if side_enum == OrderSide.BUY:
            self._cash -= (trade_amount + commission)
            pos = self._positions.get(code, {"quantity": 0, "avg_cost": Decimal("0")})
            total_cost = pos["avg_cost"] * Decimal(str(pos["quantity"])) + trade_amount
            pos["quantity"] = int(pos["quantity"]) + qty
            pos["avg_cost"] = total_cost / Decimal(str(pos["quantity"])) if pos["quantity"] > 0 else Decimal("0")
            self._positions[code] = pos
            self._today_buys[code] = self._today_buys.get(code, 0) + qty
        else:
            self._cash += (trade_amount - commission - tax)
            pos = self._positions.get(code)
            if pos:
                pos["quantity"] = int(pos["quantity"]) - qty
                if pos["quantity"] <= 0:
                    self._positions.pop(code, None)
                else:
                    self._positions[code] = pos

        self._trade_history.append({
            "order_id": order.order_id, "code": code, "side": side,
            "quantity": qty, "price": fill_price, "amount": trade_amount,
            "commission": commission + tax,
            "time": order.updated_at,
        })

        return order

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单."""
        order = self._orders.get(order_id)
        if order is None or order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def get_positions(self) -> list[Position]:
        """查询持仓."""
        result = []
        for code, pos in self._positions.items():
            qty = pos["quantity"]
            if qty <= 0:
                continue
            result.append(Position(
                code=code,
                quantity=qty,
                avg_cost=pos["avg_cost"],
                market_value=pos["avg_cost"] * Decimal(str(qty)),  # stub: 用成本价
                unrealized_pnl=Decimal("0"),
            ))
        return result

    def get_account(self) -> AccountInfo:
        """查询账户."""
        mv = sum(
            p["avg_cost"] * Decimal(str(p["quantity"]))
            for p in self._positions.values()
        )
        total = self._cash + mv
        return AccountInfo(
            account_id=self._account_id,
            total_equity=total,
            cash=self._cash,
            market_value=mv,
            frozen_cash=self._frozen_cash,
            available_cash=self._cash - self._frozen_cash,
            total_pnl=total - self._initial_cash,
            daily_pnl=Decimal("0"),
        )

    def get_order(self, order_id: str) -> BrokerOrder | None:
        """查询订单."""
        return self._orders.get(order_id)

    def get_orders(self, status_filter: list[str] | None = None) -> list[BrokerOrder]:
        """查询订单列表."""
        if status_filter is None:
            s_set = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL}
        else:
            s_set = {OrderStatus(s) for s in status_filter}
        return [o for o in self._orders.values() if o.status in s_set]

    # ── Stub 特有方法 ─────────────────────────────────────

    def end_of_day(self) -> None:
        """日终清算: T+1 限制解除."""
        self._today_buys.clear()

    def reset(self) -> None:
        """重置到初始状态."""
        import uuid
        self._account_id = f"STUB_{uuid.uuid4().hex[:8]}"
        self._cash = self._initial_cash
        self._frozen_cash = Decimal("0")
        self._positions.clear()
        self._orders.clear()
        self._today_buys.clear()
        self._trade_history.clear()

    def update_market_prices(self, prices: dict[str, Decimal]) -> None:
        """更新市价 (用于计算市值和浮盈).

        Args:
            prices: {code: current_price}
        """
        for code, price in prices.items():
            if code in self._positions:
                self._positions[code]["market_price"] = price

    def total_equity(self, prices: dict[str, Decimal]) -> Decimal:
        """计算总权益 (基于市价).

        Args:
            prices: {code: current_price}

        Returns:
            总权益
        """
        mv = Decimal("0")
        for code, pos in self._positions.items():
            px = prices.get(code, pos.get("market_price", pos["avg_cost"]))
            mv += px * Decimal(str(pos["quantity"]))
        return self._cash + mv

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def trade_history(self) -> list[dict]:
        return list(self._trade_history)

    @property
    def account_id(self) -> str:
        return self._account_id

    def _get_fill_price(self, order: BrokerOrder) -> Decimal:
        """计算成交价 (含滑点)."""
        if order.side == OrderSide.BUY:
            return order.price * (Decimal("1") + self._slippage)
        return order.price * (Decimal("1") - self._slippage)
