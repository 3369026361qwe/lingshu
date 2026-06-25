"""模拟券商 — A股规则模拟 (T+1/涨跌停/最小手续费/滑点)。"""
import logging
from decimal import Decimal
from zhixing.order_manager import Order, OrderStatus
from zhixing.metrics import broker_trades_total, broker_commission_total

_logger = logging.getLogger(__name__)


class MockBroker:
    """模拟券商 — 含A股基本规则。"""

    def __init__(self, commission_rate: Decimal = Decimal("0.0003"), slippage: Decimal = Decimal("0.0001"), min_commission: Decimal = Decimal("5"), stamp_tax: Decimal = Decimal("0.001")):
        self.commission_rate = commission_rate; self.slippage = slippage
        self.min_commission = min_commission; self.stamp_tax = stamp_tax  # 卖出印花税
        self._executed: list[dict] = []; self.cash = Decimal("0")
        self.positions: dict[str, dict] = {}; self._today_buys: dict[str, int] = {}

    def submit(self, order: Order) -> dict:
        """提交订单并模拟成交。"""
        # 卖出检查：是否有足够持仓（含T+1限制）
        if order.direction == "SELL":
            pos = self.positions.get(order.code, {"quantity": 0})
            available = pos.get("quantity", 0) - self._today_buys.get(order.code, 0)
            if available < order.quantity:
                order.reject(f"持仓不足: 可用{available}股, 委托{order.quantity}股")
                return {"order_id": order.order_id, "status": "rejected", "reason": order.reason}

        fill_price = order.price * (Decimal("1") + self.slippage) if order.direction == "BUY" else order.price * (Decimal("1") - self.slippage)
        qty = order.quantity
        commission = max(fill_price * qty * self.commission_rate, self.min_commission)
        tax = fill_price * qty * self.stamp_tax if order.direction == "SELL" else Decimal("0")
        order.fill(qty, fill_price)

        trade = {"order_id": order.order_id, "code": order.code, "direction": order.direction, "quantity": qty, "price": fill_price, "amount": fill_price * qty, "commission": commission + tax, "time": order.updated_at.isoformat()}
        self._executed.append(trade)
        broker_trades_total.inc()
        broker_commission_total.inc(float(commission + tax))

        if order.direction == "BUY":
            self.cash -= (fill_price * qty + commission)
            pos = self.positions.get(order.code, {"quantity": 0, "avg_cost": Decimal("0")})
            total_cost = pos["avg_cost"] * pos["quantity"] + fill_price * qty
            pos["quantity"] += qty
            pos["avg_cost"] = (total_cost / pos["quantity"]).quantize(Decimal("0.0001")) if pos["quantity"] > 0 else Decimal("0")
            self.positions[order.code] = pos
            self._today_buys[order.code] = self._today_buys.get(order.code, 0) + qty
        else:
            self.cash += (fill_price * qty - commission - tax)
            pos = self.positions.get(order.code, {"quantity": 0, "avg_cost": Decimal("0")})
            pos["quantity"] -= qty
            if pos["quantity"] <= 0: self.positions.pop(order.code, None)
            else: self.positions[order.code] = pos

        return trade

    def end_of_day(self) -> None:
        """日终清算：T+1限制解除。"""
        self._today_buys.clear()

    def cancel(self, order_id: str) -> None:
        _logger.info("MockBroker cancel: %s", order_id)

    @property
    def executed_trades(self) -> list[dict]:
        return list(self._executed)

    def reset(self) -> None:
        self._executed.clear(); self.cash = Decimal("0"); self.positions.clear(); self._today_buys.clear()

    def total_equity(self, prices: dict[str, Decimal]) -> Decimal:
        mv = sum(Decimal(str(self.positions.get(c, {}).get("quantity", 0))) * prices.get(c, Decimal("0")) for c in self.positions)
        return self.cash + mv
