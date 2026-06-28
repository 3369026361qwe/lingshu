"""风控总入口 — 串联 L1-L3 检查，输出统一风控报告。"""
from decimal import Decimal

from fengkong.circuit_breaker import CircuitBreaker
from fengkong.position_limiter import PositionLimiter
from fengkong.position_tracker import PositionTracker
from fengkong.rate_limiter import RateLimiter
from fengkong.stress_tester import StressTester
from fengkong.var_calculator import VaRCalculator


class RiskManager:
    """风控总入口 — 串联全部检查。"""

    def __init__(self):
        self.breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.position_limiter = PositionLimiter()
        self.tracker = PositionTracker()
        self.stress_tester = StressTester()

    def check_all(self, portfolio: list[dict], daily_pnl: Decimal, current_equity: Decimal, daily_returns: list[Decimal], industry_map: dict[str, str] | None = None) -> dict:
        """执行全部风控检查。

        Returns:
            {passed, breaker_state, position_check, var_report, stress_results, risk_level, advice}
        """
        # L1: 熔断器
        breaker_state = self.breaker.update(daily_pnl, current_equity)
        blocked = self.breaker.is_blocked

        # L1: 频率限制
        rate_ok = self.rate_limiter.acquire()

        # L2: 仓位检查
        pos_check = self.position_limiter.check(portfolio, industry_map)

        # L3: VaR
        var_report = VaRCalculator.calculate_all(daily_returns, current_equity)

        # L3: 压力测试
        positions_dict = {r["code"]: {"market_value": float(r.get("weight", 0)) * float(current_equity), "beta": 1.0} for r in portfolio}
        stress = self.stress_tester.run_all(current_equity, positions_dict)

        # 综合风险等级
        risk_score = 1
        if blocked:
            risk_score = 5
        elif not pos_check["passed"]:
            risk_score = 3
        elif var_report.get("var_amount") and var_report["var_amount"] > current_equity * Decimal("0.05"):
            risk_score = 2

        levels = {1: "LOW", 2: "GUARDED", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}

        return {"passed": not blocked and pos_check["passed"] and rate_ok, "breaker_state": breaker_state.value, "blocked": blocked, "rate_limited": not rate_ok, "position_check": pos_check, "var_report": var_report, "stress_results": stress, "risk_level": levels.get(risk_score, "LOW"), "risk_score": risk_score, "advice": "禁止交易" if blocked else ("需调整仓位" if not pos_check["passed"] else "可正常交易")}
