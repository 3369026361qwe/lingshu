"""订单管理器 — 订单生命周期（创建→提交→成交→取消）+ A股规则。"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from zhixing.metrics import orders_created, orders_filled, orders_rejected

# A股交易规则
A_SHARE_LOT_SIZE = 100  # 每手100股


class OrderStatus(str, Enum):
    PENDING = "PENDING"; SUBMITTED = "SUBMITTED"; FILLED = "FILLED"; PARTIAL = "PARTIAL"; CANCELLED = "CANCELLED"; REJECTED = "REJECTED"


class Order:
    """交易订单。"""
    def __init__(self, code: str, direction: str, quantity: int, price: Decimal, order_type: str = "LIMIT", reason: str = ""):
        self.order_id = uuid.uuid4().hex[:16]
        self.code = code; self.direction = direction; self.quantity = quantity; self.price = price
        self.order_type = order_type; self.status = OrderStatus.PENDING; self.filled_qty = 0
        self.filled_avg_price: Decimal | None = None; self.reason = reason
        self.created_at = datetime.now(timezone.utc); self.updated_at = self.created_at

    def fill(self, qty: int, price: Decimal) -> None:
        """成交（支持多次部分成交，价格加权平均）。"""
        if self.filled_avg_price is not None and self.filled_qty > 0:
            total_value = self.filled_avg_price * self.filled_qty + price * qty
            self.filled_qty += qty
            self.filled_avg_price = total_value / self.filled_qty
        else:
            self.filled_qty = qty
            self.filled_avg_price = price
        self.status = OrderStatus.FILLED if self.filled_qty >= self.quantity else OrderStatus.PARTIAL
        self.updated_at = datetime.now(timezone.utc)

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED; self.updated_at = datetime.now(timezone.utc)

    def reject(self, reason: str = "") -> None:
        self.status = OrderStatus.REJECTED; self.reason = reason; self.updated_at = datetime.now(timezone.utc)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)

    @property
    def is_done(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


class OrderManager:
    """订单生命周期管理器。"""

    def __init__(self):
        self._orders: dict[str, Order] = {}
        self._history: list[Order] = []
        # T+1 追踪：今日买入的股票
        self._today_buys: dict[str, int] = {}  # {code: quantity}

    def create(self, code: str, direction: str, quantity: int, price: Decimal, order_type: str = "LIMIT", reason: str = "") -> Order:
        if direction not in ("BUY", "SELL"): raise ValueError(f"Invalid direction: {direction}")
        if quantity <= 0: raise ValueError(f"Invalid quantity: {quantity}")
        if price <= 0: raise ValueError(f"Invalid price: {price}")
        # A股：按手取整（100股/手）
        qty = (quantity // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
        if qty == 0: raise ValueError(f"Quantity too small, minimum {A_SHARE_LOT_SIZE} shares")
        order = Order(code, direction, qty, price, order_type, reason)
        self._orders[order.order_id] = order
        orders_created.labels(direction=direction).inc()
        return order

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def fill(self, order_id: str, qty: int, price: Decimal) -> Order:
        order = self._orders[order_id]
        if order.is_done:
            raise ValueError(f"Cannot fill {order.status.value} order {order_id}")
        order.fill(qty, price)
        orders_filled.labels(direction=order.direction).inc()
        if order.status == OrderStatus.FILLED and order.direction == "BUY":
            self._today_buys[order.code] = self._today_buys.get(order.code, 0) + qty
        self._history.append(order)
        return order

    def cancel(self, order_id: str, reason: str = "") -> Order:
        order = self._orders[order_id]
        if order.is_done:
            raise ValueError(f"Cannot cancel {order.status.value} order {order_id}")
        order.cancel()
        if reason: order.reason = reason
        self._history.append(order)
        return order

    def reject(self, order_id: str, reason: str = "") -> Order:
        order = self._orders[order_id]
        order.reject(reason)
        orders_rejected.inc()
        self._history.append(order)
        return order

    def can_sell(self, code: str, quantity: int, available_qty: int) -> bool:
        """T+1检查：今日买入的股票不可卖出。"""
        today_bought = self._today_buys.get(code, 0)
        return (available_qty - today_bought) >= quantity

    @property
    def active_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.is_active]

    @property
    def all_orders(self) -> list[Order]:
        return list(self._orders.values())

    @property
    def history(self) -> list[Order]:
        return list(self._history)
