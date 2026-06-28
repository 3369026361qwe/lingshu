"""测试风控模块: 熔断器/频率限制/仓位限制/持仓追踪/VaR/压力测试/风控总入口。"""
from decimal import Decimal

from fengkong.circuit_breaker import BreakerState, CircuitBreaker
from fengkong.position_limiter import PositionLimiter
from fengkong.position_tracker import PositionTracker
from fengkong.rate_limiter import RateLimiter
from fengkong.risk_manager import RiskManager
from fengkong.stress_tester import StressTester
from fengkong.var_calculator import VaRCalculator


class TestCircuitBreaker:
    def test_normal_flow(self):
        cb = CircuitBreaker()
        for _ in range(5):
            s = cb.update(Decimal("0.01"), Decimal("1100000"))
            assert s == BreakerState.CLOSED

    def test_consecutive_losses_open(self):
        cb = CircuitBreaker(max_consecutive_losses=3)
        for _ in range(3):
            cb.update(Decimal("-50000"), Decimal("1000000"))
        assert cb.state == BreakerState.OPEN

    def test_drawdown_trigger(self):
        cb = CircuitBreaker(max_drawdown=Decimal("0.10"))
        cb.update(Decimal("0"), Decimal("1000000"))
        cb.update(Decimal("0"), Decimal("850000"))  # -15%
        assert cb.state == BreakerState.OPEN

    def test_cooldown_to_half_open(self):
        cb = CircuitBreaker(max_consecutive_losses=1, cooldown_seconds=1)
        cb.update(Decimal("-1"), Decimal("1000000"))
        assert cb.state == BreakerState.OPEN
        import time
        time.sleep(1.1)
        assert cb.state == BreakerState.HALF_OPEN

    def test_reset(self):
        cb = CircuitBreaker(max_consecutive_losses=1)
        cb.update(Decimal("-1"), Decimal("1000000"))
        cb.reset()
        assert cb.state == BreakerState.CLOSED

    def test_is_blocked(self):
        cb = CircuitBreaker(max_consecutive_losses=1)
        cb.update(Decimal("-1"), Decimal("1000000"))
        assert cb.is_blocked


class TestRateLimiter:
    def test_acquire_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.acquire()

    def test_acquire_exceeded(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.acquire()
        assert not rl.acquire()

    def test_remaining(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.remaining == 5
        rl.acquire()
        assert rl.remaining == 4


class TestPositionLimiter:
    def test_all_pass(self):
        pl = PositionLimiter()
        portfolio = [{"code": "000001", "weight": Decimal("0.08")}, {"code": "000002", "weight": Decimal("0.06")}]
        assert pl.check(portfolio)["passed"]

    def test_total_exceeded(self):
        pl = PositionLimiter()
        portfolio = [{"code": "000001", "weight": Decimal("0.50")}, {"code": "000002", "weight": Decimal("0.50")}]
        assert not pl.check(portfolio)["passed"]

    def test_single_exceeded(self):
        pl = PositionLimiter(max_single=Decimal("0.10"))
        portfolio = [{"code": "000001", "weight": Decimal("0.15")}]
        assert not pl.check(portfolio)["passed"]

    def test_industry_exceeded(self):
        pl = PositionLimiter(max_industry=Decimal("0.30"))
        portfolio = [{"code": "000001", "weight": Decimal("0.20")}, {"code": "000002", "weight": Decimal("0.15")}]
        ind = {"000001": "银行", "000002": "银行"}
        assert not pl.check(portfolio, ind)["passed"]

    def test_kelly(self):
        f = PositionLimiter().calc_kelly(Decimal("0.6"), Decimal("0.02"), Decimal("0.01"))
        assert f is not None and f > 0


class TestPositionTracker:
    def test_update_and_get(self):
        pt = PositionTracker()
        pt.update("000001", 1000, Decimal("10"), Decimal("10.5"), "银行")
        pos = pt.get("000001")
        assert pos["quantity"] == 1000
        assert pos["market_value"] == Decimal("10500")

    def test_snapshot(self):
        pt = PositionTracker()
        pt.update("000001", 1000, Decimal("10"), Decimal("10"))
        snap = pt.snapshot(Decimal("100000"))
        assert len(snap) == 1
        assert snap[0]["weight"] == Decimal("0.1")

    def test_remove(self):
        pt = PositionTracker()
        pt.update("000001", 100, Decimal("10"))
        pt.remove("000001")
        assert pt.get("000001") is None


class TestVaRCalculator:
    def test_historical_var(self):
        returns = [Decimal(str(0.001 * (i % 7 - 3))) for i in range(252)]
        var = VaRCalculator.historical_var(returns)
        assert var is not None and var > 0

    def test_cvar_greater_than_var(self):
        returns = [Decimal(str(0.001 * (i % 7 - 3))) for i in range(252)]
        var = VaRCalculator.historical_var(returns)
        cvar = VaRCalculator.historical_cvar(returns)
        assert cvar >= var

    def test_parametric_var(self):
        returns = [Decimal(str(0.001 * (i % 7 - 3))) for i in range(252)]
        pvar = VaRCalculator.parametric_var(returns)
        assert pvar is not None

    def test_insufficient_data(self):
        assert VaRCalculator.historical_var([Decimal("0.01")] * 5) is None

    def test_calculate_all(self):
        returns = [Decimal(str(0.001 * (i % 7 - 3))) for i in range(100)]
        report = VaRCalculator.calculate_all(returns, Decimal("1000000"))
        assert "var_95" in report and "cvar_95" in report


class TestStressTester:
    def test_run_scenario(self):
        positions = {"000001": {"market_value": 500000, "beta": 1.2}}
        result = StressTester.run_scenario(Decimal("1000000"), positions, "2008_financial_crisis")
        assert result["estimated_loss"] < 0

    def test_run_all(self):
        positions = {"000001": {"market_value": 500000, "beta": 1.0}}
        results = StressTester.run_all(Decimal("1000000"), positions)
        assert len(results) == 4


class TestRiskManager:
    def test_check_all_normal(self):
        rm = RiskManager()
        portfolio = [{"code": "000001", "weight": Decimal("0.08")}]
        returns = [Decimal("0.001")] * 100
        result = rm.check_all(portfolio, Decimal("1000"), Decimal("1000000"), returns)
        assert result["passed"]
        assert result["risk_level"] == "LOW"

    def test_check_all_blocked(self):
        rm = RiskManager()
        rm.breaker.max_consecutive_losses = 1
        portfolio = [{"code": "000001", "weight": Decimal("0.05")}]
        returns = [Decimal("0.001")] * 100
        result = rm.check_all(portfolio, Decimal("-100000"), Decimal("1000000"), returns)
        assert result["blocked"]
        assert result["risk_level"] == "CRITICAL"
