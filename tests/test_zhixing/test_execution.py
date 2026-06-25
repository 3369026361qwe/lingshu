"""测试交易执行: 订单管理/批量执行/模拟券商/成交记录。"""
from decimal import Decimal
import pytest
from zhixing.order_manager import OrderManager, OrderStatus
from zhixing.batch_executor import BatchExecutor
from zhixing.mock_broker import MockBroker
from zhixing.trade_recorder import TradeRecorder


class TestOrderManager:
    def test_create_order(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.50"))
        assert o.code == "000001" and o.status == OrderStatus.PENDING

    def test_fill_order(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.50"))
        om.fill(o.order_id, 1000, Decimal("10.48"))
        assert o.status == OrderStatus.FILLED and o.filled_qty == 1000

    def test_partial_fill(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.50"))
        om.fill(o.order_id, 500, Decimal("10.50"))
        assert o.status == OrderStatus.PARTIAL

    def test_cancel(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.50"))
        om.cancel(o.order_id)
        assert o.status == OrderStatus.CANCELLED

    def test_reject(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.50"))
        om.reject(o.order_id, "涨跌停")
        assert o.status == OrderStatus.REJECTED

    def test_active_orders(self):
        om = OrderManager()
        om.create("000001", "BUY", 1000, Decimal("10"))
        om.create("000002", "SELL", 500, Decimal("20"))
        assert len(om.active_orders) == 2

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            OrderManager().create("000001", "HOLD", 100, Decimal("10"))

    def test_invalid_quantity(self):
        with pytest.raises(ValueError):
            OrderManager().create("000001", "BUY", 0, Decimal("10"))

    def test_lot_size_rounding(self):
        """A股按手取整：150股→100股。"""
        o = OrderManager().create("000001", "BUY", 150, Decimal("10"))
        assert o.quantity == 100

    def test_too_small_quantity(self):
        """小于1手的委托应拒绝。"""
        with pytest.raises(ValueError):
            OrderManager().create("000001", "BUY", 50, Decimal("10"))

    def test_cannot_fill_done_order(self):
        om = OrderManager()
        o = om.create("000001", "BUY", 100, Decimal("10"))
        om.fill(o.order_id, 100, Decimal("10"))
        with pytest.raises(ValueError):
            om.fill(o.order_id, 100, Decimal("10"))  # 已成交

    def test_partial_fill_avg_price(self):
        """部分成交后价格应为加权平均。"""
        om = OrderManager()
        o = om.create("000001", "BUY", 200, Decimal("10"))
        om.fill(o.order_id, 100, Decimal("10.00"))
        om.fill(o.order_id, 100, Decimal("12.00"))
        assert o.filled_avg_price == Decimal("11.00")

    def test_t1_cannot_sell(self):
        """T+1: 今日买入的股票不能卖出。"""
        om = OrderManager()
        bo = om.create("000001", "BUY", 100, Decimal("10"))
        om.fill(bo.order_id, 100, Decimal("10"))
        # 可用持仓100但今日买入100→可卖0
        assert not om.can_sell("000001", 100, 100)


class TestMockBroker:
    def test_submit_buy(self):
        broker = MockBroker()
        om = OrderManager()
        o = om.create("000001", "BUY", 1000, Decimal("10.00"))
        trade = broker.submit(o)
        assert o.status == OrderStatus.FILLED
        assert trade["commission"] >= Decimal("5")
        assert broker.positions["000001"]["quantity"] == 1000

    def test_submit_sell(self):
        broker = MockBroker()
        broker.positions["000001"] = {"quantity": 1000, "avg_cost": Decimal("10")}
        broker.end_of_day()  # 模拟持仓非今日买入
        om = OrderManager()
        o = om.create("000001", "SELL", 500, Decimal("11.00"))
        broker.submit(o)
        assert broker.positions["000001"]["quantity"] == 500

    def test_slippage(self):
        broker = MockBroker(slippage=Decimal("0.01"))
        om = OrderManager()
        o = om.create("000001", "BUY", 100, Decimal("10.00"))
        trade = broker.submit(o)
        assert trade["price"] > Decimal("10.00")

    def test_reset(self):
        broker = MockBroker()
        om = OrderManager()
        broker.submit(om.create("000001", "BUY", 100, Decimal("10")))
        broker.reset()
        assert len(broker.executed_trades) == 0


class TestBatchExecutor:
    def test_execute_rebalance(self):
        broker = MockBroker()
        om = OrderManager()
        executor = BatchExecutor(om, broker)
        trades = {"buys": [{"code": "000001", "amount": Decimal("50000")}], "sells": []}
        orders = executor.execute_rebalance(trades, {"000001": Decimal("10.00")})
        assert len(orders) == 1


class TestTradeRecorder:
    def test_record(self):
        rec = TradeRecorder()
        om = OrderManager()
        o = om.create("000001", "BUY", 100, Decimal("10"))
        om.fill(o.order_id, 100, Decimal("10.00"))
        trade = rec.record(o, Decimal("10.00"))
        assert trade["code"] == "000001"
        assert rec.total_commission == Decimal("0")

    def test_multiple_trades(self):
        rec = TradeRecorder()
        om = OrderManager()
        for i in range(3):
            o = om.create(f"00000{i+1}", "BUY", 100, Decimal("10"))
            om.fill(o.order_id, 100, Decimal("10.00"))
            rec.record(o, Decimal("10.00"), Decimal("5"))
        assert len(rec.trades) == 3
        assert rec.total_commission == Decimal("15")
