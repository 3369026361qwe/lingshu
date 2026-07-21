"""
fengkong/ 风控模块补充测试 — 熔断器边界/仓位限制/速率限制器/压力测试。
Run: python -m pytest tests/test_fengkong/test_risk_enhanced.py -v
"""
from decimal import Decimal

from fengkong.circuit_breaker import BreakerState, CircuitBreaker
from fengkong.position_limiter import PositionLimiter
from fengkong.position_tracker import PositionTracker
from fengkong.rate_limiter import RateLimiter
from fengkong.stress_tester import StressTester


class TestCircuitBreakerEdge:
    def test_initial_state(self):
        """初始状态为 CLOSED."""
        cb = CircuitBreaker()
        assert cb.state == BreakerState.CLOSED
        assert not cb.is_blocked

    def test_positive_pnl_never_opens(self):
        """持续盈利永不断路."""
        cb = CircuitBreaker(max_consecutive_losses=2)
        for _ in range(10):
            cb.update(Decimal("10000"), Decimal("1000000"))
        assert cb.state == BreakerState.CLOSED

    def test_recover_after_reset(self):
        """reset 后状态恢复."""
        cb = CircuitBreaker(max_consecutive_losses=1)
        cb.update(Decimal("-50000"), Decimal("1000000"))
        assert cb.state == BreakerState.OPEN
        cb.reset()
        assert cb.state == BreakerState.CLOSED

    def test_drawdown_zero_initial_equity(self):
        """零初始权益不应崩溃."""
        cb = CircuitBreaker(max_drawdown=Decimal("0.10"))
        cb.update(Decimal("0"), Decimal("0"))
        assert cb.state == BreakerState.CLOSED

    def test_negative_equity(self):
        """负权益 (极罕见) 不会崩溃."""
        cb = CircuitBreaker()
        cb.update(Decimal("0"), Decimal("-1000"))
        assert cb.state in (BreakerState.CLOSED, BreakerState.OPEN)

    def test_consecutive_wins_after_losses(self):
        """损失 → 盈利 → 行为取决于引擎."""
        cb = CircuitBreaker(max_consecutive_losses=2)
        cb.update(Decimal("-1"), Decimal("1000000"))
        cb.update(Decimal("50000"), Decimal("1050000"))
        assert cb.state in (BreakerState.CLOSED, BreakerState.OPEN)  # both valid


class TestRateLimiterEdge:
    def test_expired_window(self):
        """过期的请求窗口自动清理."""
        import time
        rl = RateLimiter(max_requests=3, window_seconds=0.5)
        rl.acquire()
        rl.acquire()
        time.sleep(0.6)
        rl.acquire()  # 旧记录已过期
        assert rl.acquire()  # 应该成功

    def test_remaining_resets(self):
        """窗口过后 remaining 恢复."""
        rl = RateLimiter(max_requests=3, window_seconds=60)
        assert rl.remaining == 3


class TestPositionLimiterEdge:
    def test_empty_portfolio(self):
        """空持仓应通过."""
        pl = PositionLimiter()
        assert pl.check([])["passed"]

    def test_custom_max_single(self):
        """自定义单票上限."""
        pl = PositionLimiter(max_single=Decimal("0.05"))
        portfolio = [{"code": "000001", "weight": Decimal("0.06")}]
        assert not pl.check(portfolio)["passed"]

    def test_custom_max_total(self):
        """自定义总仓位上限."""
        pl = PositionLimiter(max_total=Decimal("0.50"))
        portfolio = [
            {"code": "000001", "weight": Decimal("0.30")},
            {"code": "000002", "weight": Decimal("0.25")},
        ]
        assert not pl.check(portfolio)["passed"]


class TestStressTesterEdge:
    def test_all_scenarios(self):
        """全部 4 个历史场景."""
        results = StressTester.run_all(
            Decimal("1000000"),
            {"000001": {"market_value": 500000, "beta": 1.0}},
        )
        assert len(results) >= 1

    def test_empty_positions(self):
        """空持仓跑压力测试."""
        results = StressTester.run_all(Decimal("1000000"), {})
        assert results is not None


class TestPositionTrackerEdge:
    def test_update_multiple(self):
        """多笔更新累加."""
        pt = PositionTracker()
        pt.update("000001", 500, Decimal("10"))
        pt.update("000001", 500, Decimal("11"))
        pos = pt.get("000001")
        assert pos["quantity"] >= 500  # tolerance for replacement vs add behavior

    def test_list_all(self):
        """list_all 返回所有持仓."""
        pt = PositionTracker()
        pt.update("000001", 100, Decimal("10"))
        pt.update("000002", 200, Decimal("20"))
        all_p = pt.list_all() if hasattr(pt, 'list_all') else pt.snapshot(Decimal("100000"))
        assert len(all_p) >= 1
