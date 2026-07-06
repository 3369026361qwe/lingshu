"""测试新模块: market_impact.py 和 live_broker.py."""
from decimal import Decimal

from zhixing.live_broker import (
    AbstractBroker,
    AccountInfo,
    BrokerOrder,
    OrderSide,
    OrderType,
    Position,
    StubBroker,
)
from zhixing.live_broker import (
    OrderStatus as BrokerOrderStatus,
)
from zhixing.market_impact import MarketImpactModel

# ── MarketImpactModel 测试 ────────────────────────────

class TestMarketImpactAlmgrenChriss:
    """Almgren-Chriss 冲击模型测试."""

    def test_basic_estimate(self):
        """基本估计: 10万股, 日均100万股, 波动率2%, 价格10元."""
        impact = MarketImpactModel.almgren_chriss(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        assert impact.total_impact > Decimal("0")
        assert impact.permanent_impact > Decimal("0")
        assert impact.temporary_impact > Decimal("0")
        assert impact.participation_rate > Decimal("0")
        assert impact.effective_price > impact.arrival_price
        assert impact.total_cost > Decimal("0")

    def test_total_is_sum(self):
        """总冲击 = 永久 + 瞬时."""
        impact = MarketImpactModel.almgren_chriss(
            order_quantity=50_000,
            daily_volume=500_000,
            daily_volatility=Decimal("0.01"),
            price=Decimal("20"),
        )
        assert abs(impact.total_impact - (impact.permanent_impact + impact.temporary_impact)) < Decimal("0.01")

    def test_zero_volume(self):
        """零订单量 → 零冲击."""
        impact = MarketImpactModel.almgren_chriss(
            order_quantity=0,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        assert impact.total_impact == Decimal("0")
        assert impact.total_cost == Decimal("0")

    def test_participation_rate(self):
        """参与率 = 订单量 / 日均成交量."""
        impact = MarketImpactModel.almgren_chriss(
            order_quantity=200_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        assert float(impact.participation_rate) == 0.2

    def test_larger_order_higher_impact(self):
        """订单越大, 冲击越大."""
        small = MarketImpactModel.almgren_chriss(
            order_quantity=10_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        large = MarketImpactModel.almgren_chriss(
            order_quantity=500_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        assert large.total_impact > small.total_impact

    def test_higher_volatility_higher_impact(self):
        """波动率越高, 冲击越大."""
        low_vol = MarketImpactModel.almgren_chriss(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.01"),
            price=Decimal("10"),
        )
        high_vol = MarketImpactModel.almgren_chriss(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.05"),
            price=Decimal("10"),
        )
        assert high_vol.total_impact > low_vol.total_impact

    def test_custom_parameters(self):
        """自定义 gamma/eta."""
        impact = MarketImpactModel.almgren_chriss(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
            gamma=Decimal("0.20"),
            eta=Decimal("0.30"),
        )
        assert impact.total_impact > Decimal("0")


class TestMarketImpactSqrtLiquidity:
    """平方根流动性模型测试."""

    def test_basic_estimate(self):
        """基本估计."""
        impact = MarketImpactModel.sqrt_liquidity(
            order_quantity=100_000,
            daily_volume=1_000_000,
            price=Decimal("10"),
        )
        assert impact.total_impact > Decimal("0")
        assert impact.permanent_impact == Decimal("0")  # sqrt 模型无永久冲击
        assert impact.temporary_impact > Decimal("0")

    def test_participation_10pct(self):
        """参与率 10% → 冲击 ~ κ * sqrt(0.1)."""
        impact = MarketImpactModel.sqrt_liquidity(
            order_quantity=100_000,
            daily_volume=1_000_000,
            price=Decimal("10"),
            kappa=Decimal("20"),
        )
        # impact ≈ 20 * sqrt(0.1) ≈ 6.32 bps
        expected = Decimal("20") * Decimal("0.1").sqrt()
        assert abs(impact.total_impact - expected) < Decimal("1")

    def test_zero_participation(self):
        """零参与率 → 零冲击."""
        impact = MarketImpactModel.sqrt_liquidity(
            order_quantity=0,
            daily_volume=1_000_000,
            price=Decimal("10"),
        )
        assert impact.total_impact == Decimal("0")

    def test_custom_kappa(self):
        """自定义 kappa 参数."""
        impact = MarketImpactModel.sqrt_liquidity(
            order_quantity=100_000,
            daily_volume=1_000_000,
            price=Decimal("10"),
            kappa=Decimal("50"),
        )
        assert impact.total_impact > Decimal("10")


class TestMarketImpactEstimate:
    """统一接口 estimate() 测试."""

    def test_default_model(self):
        """默认使用 Almgren-Chriss."""
        impact = MarketImpactModel.estimate(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
        )
        assert impact.permanent_impact > Decimal("0")

    def test_sqrt_model(self):
        """指定 sqrt_liquidity 模型."""
        impact = MarketImpactModel.estimate(
            order_quantity=100_000,
            daily_volume=1_000_000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
            model="sqrt_liquidity",
        )
        assert impact.permanent_impact == Decimal("0")


class TestExecutionSchedules:
    """TWAP/VWAP 执行调度测试."""

    def test_twap_slices(self):
        """TWAP 切片数量正确."""
        slices = MarketImpactModel.optimal_twap_slices(
            order_quantity=10000,
            daily_volume=1000000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
            trading_minutes=240,
            slice_interval_minutes=15,
        )
        assert len(slices) == 16  # 240/15 = 16
        for s in slices:
            assert s["quantity"] >= 100  # A股按手
            assert s["expected_impact_bps"] >= 0
            assert s["expected_cost"] >= 0

    def test_twap_single_slice(self):
        """单个切片 (间隔 > 交易时间)."""
        slices = MarketImpactModel.optimal_twap_slices(
            order_quantity=10000,
            daily_volume=1000000,
            daily_volatility=Decimal("0.02"),
            price=Decimal("10"),
            trading_minutes=60,
            slice_interval_minutes=120,
        )
        assert len(slices) == 1

    def test_vwap_slices(self):
        """VWAP 切片 — 均匀成交量分布."""
        slices = MarketImpactModel.optimal_vwap_slices(
            order_quantity=10000,
            daily_volume=1000000,
            price=Decimal("10"),
        )
        assert len(slices) == 16
        total_qty = sum(s["quantity"] for s in slices)
        assert total_qty >= 10000 - 1600  # 允许取整误差

    def test_vwap_custom_profile(self):
        """自定义成交量分布."""
        profile = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1,
                   0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05]
        slices = MarketImpactModel.optimal_vwap_slices(
            order_quantity=10000,
            daily_volume=1000000,
            price=Decimal("10"),
            volume_profile=profile,
        )
        assert len(slices) == 16


# ── StubBroker (live_broker) 测试 ─────────────────────

class TestStubBroker:
    """StubBroker 实盘接口测试."""

    def test_create_broker(self):
        """创建 broker."""
        broker = StubBroker(initial_cash=Decimal("1000000"))
        assert broker.cash == Decimal("1000000")
        assert len(broker.account_id) > 0

    def test_submit_buy_order(self):
        """提交买单."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        assert order.status == BrokerOrderStatus.FILLED
        assert order.filled_qty == 1000
        assert order.filled_avg_price > Decimal("10.00")  # 含滑点
        assert broker.cash < Decimal("100000")

    def test_submit_sell_order(self):
        """提交卖单."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        # 先买入
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.end_of_day()
        # 再卖出
        order = broker.submit_order("000001", "SELL", 500, Decimal("11.00"))
        assert order.status == BrokerOrderStatus.FILLED
        # cash ≈ 100000 - buy(~10000含佣金) + sell(~5500 - 佣金 - 印花税)
        assert broker.cash > Decimal("95000")

    def test_sell_insufficient(self):
        """卖出不足: 没有持仓."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "SELL", 1000, Decimal("10.00"))
        assert order.status == BrokerOrderStatus.REJECTED

    def test_t1_restriction(self):
        """T+1: 今日买入当天不可卖出."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        # 同一天卖出 → 被拒绝
        order = broker.submit_order("000001", "SELL", 500, Decimal("11.00"))
        assert order.status == BrokerOrderStatus.REJECTED
        assert "持仓不足" in order.reason

    def test_cancel_order(self):
        """撤销订单 (已成交不可撤)."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        # Stub 立即成交, 无法撤单
        assert not broker.cancel_order(order.order_id)

    def test_get_positions(self):
        """查询持仓."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].code == "000001"
        assert positions[0].quantity == 1000

    def test_get_positions_empty(self):
        """空持仓."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        positions = broker.get_positions()
        assert len(positions) == 0

    def test_get_account(self):
        """查询账户."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        acct = broker.get_account()
        assert acct.total_equity > Decimal("0")
        assert acct.cash > Decimal("0")
        assert acct.market_value > Decimal("0")

    def test_get_order(self):
        """查询订单."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        found = broker.get_order(order.order_id)
        assert found is not None
        assert found.code == "000001"

    def test_get_order_not_found(self):
        """查询不存在的订单."""
        broker = StubBroker()
        assert broker.get_order("nonexistent") is None

    def test_get_orders(self):
        """查询活动订单列表."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.submit_order("000002", "BUY", 500, Decimal("20.00"))
        # Stub 立即成交, 活动订单列表应空
        orders = broker.get_orders()
        assert len(orders) == 0  # 全部已成交

    def test_reset(self):
        """重置到初始状态."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.reset()
        assert broker.cash == Decimal("100000")
        assert len(broker.get_positions()) == 0
        assert len(broker.trade_history) == 0

    def test_total_equity(self):
        """市价计算总权益."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        equity = broker.total_equity({"000001": Decimal("12.00")})
        assert equity > Decimal("100000")

    def test_lot_size_rounding(self):
        """A股按手取整: 150股→100股."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "BUY", 150, Decimal("10.00"))
        assert order.quantity == 100

    def test_too_small_quantity(self):
        """小于一手 → 报错."""
        import pytest
        broker = StubBroker()
        with pytest.raises(ValueError):
            broker.submit_order("000001", "BUY", 50, Decimal("10.00"))

    def test_multiple_positions(self):
        """多只股票持仓."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.submit_order("000002", "BUY", 500, Decimal("20.00"))
        positions = broker.get_positions()
        assert len(positions) == 2

    def test_sell_all(self):
        """全部卖出."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.end_of_day()
        broker.submit_order("000001", "SELL", 1000, Decimal("11.00"))
        positions = broker.get_positions()
        assert len(positions) == 0

    def test_trade_history(self):
        """成交历史."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        history = broker.trade_history
        assert len(history) == 1
        assert history[0]["code"] == "000001"

    def test_update_market_prices(self):
        """更新市价."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        broker.submit_order("000001", "BUY", 1000, Decimal("10.00"))
        broker.update_market_prices({"000001": Decimal("15.00")})
        positions = broker.get_positions()
        # market_value 用成本价计算，不随市价变化
        assert positions[0].quantity == 1000

    def test_order_types(self):
        """订单类型枚举."""
        broker = StubBroker(initial_cash=Decimal("100000"))
        order = broker.submit_order("000001", "BUY", 100, Decimal("10.00"), order_type="LIMIT")
        assert order.order_type == OrderType.LIMIT


# ── AbstractBroker 接口验证 ─────────────────────────

class TestAbstractBrokerContract:
    """验证 AbstractBroker 定义了必要的方法."""

    def test_abstract_methods(self):
        """抽象方法列表."""
        methods = [
            "submit_order", "cancel_order", "get_positions",
            "get_account", "get_order", "get_orders",
        ]
        for m in methods:
            assert hasattr(AbstractBroker, m)
            assert callable(getattr(AbstractBroker, m))

    def test_stub_implements_all(self):
        """StubBroker 实现了所有抽象方法."""
        methods = [
            "submit_order", "cancel_order", "get_positions",
            "get_account", "get_order", "get_orders",
        ]
        for m in methods:
            assert hasattr(StubBroker, m)
            assert callable(getattr(StubBroker, m))


# ── 数据类验证 ─────────────────────────────────────

class TestDataClasses:
    """验证数据类的正确性."""

    def test_broker_order_defaults(self):
        """BrokerOrder 默认值."""
        order = BrokerOrder(code="000001", side=OrderSide.BUY, quantity=100)
        assert order.order_id != ""
        assert order.status == BrokerOrderStatus.PENDING
        assert order.order_type == OrderType.LIMIT

    def test_position_defaults(self):
        """Position 默认值."""
        pos = Position(code="000001", quantity=100, avg_cost=Decimal("10.00"))
        assert pos.market_value == Decimal("0")
        assert pos.unrealized_pnl == Decimal("0")

    def test_account_info(self):
        """AccountInfo."""
        acct = AccountInfo(
            account_id="test", total_equity=Decimal("100000"),
            cash=Decimal("50000"), market_value=Decimal("50000"),
        )
        assert acct.total_equity == Decimal("100000")
        assert acct.margin_ratio == Decimal("0")
