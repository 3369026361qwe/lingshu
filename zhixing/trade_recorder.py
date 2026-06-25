"""成交记录器 — 成交记录持久化。"""
from decimal import Decimal
from zhixing.order_manager import Order


class TradeRecorder:
    """成交记录器。"""

    def __init__(self, repository=None):
        self._repo = repository
        self._trades: list[dict] = []

    def record(self, order: Order, fill_price: Decimal, commission: Decimal = Decimal("0")) -> dict:
        trade = {"order_id": order.order_id, "code": order.code, "direction": order.direction, "quantity": order.filled_qty, "price": fill_price, "amount": fill_price * order.filled_qty, "commission": commission, "time": order.updated_at.isoformat() if order.updated_at else "", "reason": order.reason}
        self._trades.append(trade)
        if self._repo:
            try:
                from shujuku.models.jiaoyi_models import Trade as TradeModel
                import uuid
                t = TradeModel(trade_id=uuid.uuid4().hex[:12], order_id=order.order_id, code=order.code, direction=order.direction, quantity=order.filled_qty, price=fill_price, amount=fill_price * order.filled_qty, commission=commission)
                self._repo.add(t)
            except Exception:
                pass
        return trade

    @property
    def trades(self) -> list[dict]:
        return list(self._trades)

    @property
    def total_commission(self) -> Decimal:
        return sum(Decimal(str(t.get("commission", 0))) for t in self._trades)

    @property
    def total_amount(self) -> Decimal:
        return sum(Decimal(str(t.get("amount", 0))) for t in self._trades)
