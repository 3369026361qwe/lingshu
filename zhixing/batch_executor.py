"""批量调仓执行器 — 先卖后买+回滚+A股取整。"""
from decimal import Decimal

from zhixing.metrics import executor_batch_total
from zhixing.order_manager import A_SHARE_LOT_SIZE, Order, OrderManager


class BatchExecutor:
    """批量调仓执行器。"""

    def __init__(self, order_manager: OrderManager, broker=None):
        self._om = order_manager; self._broker = broker

    def execute_rebalance(self, trades: dict, current_prices: dict[str, Decimal]) -> list[Order]:
        """执行调仓清单（先卖后买，A股按手取整）。"""
        orders = []
        for sell in trades.get("sells", []):
            price = current_prices.get(sell["code"])
            if price and price > 0:
                qty = int(sell["amount"] / price) if price > 0 else 0
                qty = (qty // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
                if qty > 0:
                    orders.append(self._om.create(sell["code"], "SELL", qty, price, reason="rebalance"))
        for buy in trades.get("buys", []):
            price = current_prices.get(buy["code"])
            if price and price > 0:
                qty = int(buy["amount"] / price) if price > 0 else 0
                qty = (qty // A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
                if qty > 0:
                    orders.append(self._om.create(buy["code"], "BUY", qty, price, reason="rebalance"))
        if self._broker:
            for o in orders:
                self._broker.submit(o)
        executor_batch_total.labels(status="success").inc(len(orders))
        return orders

    def execute_with_rollback(self, orders: list[Order]) -> bool:
        """执行订单，失败时回滚已成交订单。"""
        executed = []
        try:
            for o in orders:
                if self._broker: self._broker.submit(o)
                executed.append(o)
            return True
        except Exception:
            executor_batch_total.labels(status="rollback").inc()
            for o in executed:
                try:
                    if self._broker: self._broker.cancel(o.order_id)
                except Exception: pass
            return False
