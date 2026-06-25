"""VaR/CVaR 实时计算器 — L3 防护层。"""
from decimal import Decimal
from typing import Optional


class VaRCalculator:
    """风险价值计算器。"""

    @staticmethod
    def historical_var(returns: list[Decimal], confidence: Decimal = Decimal("0.95")) -> Optional[Decimal]:
        """历史模拟法 VaR。"""
        if len(returns) < 20:
            return None
        sorted_returns = sorted(returns)
        idx = int(len(sorted_returns) * (Decimal("1") - confidence))
        return abs(sorted_returns[max(0, idx)])

    @staticmethod
    def historical_cvar(returns: list[Decimal], confidence: Decimal = Decimal("0.95")) -> Optional[Decimal]:
        """CVaR (Expected Shortfall)。"""
        if len(returns) < 20:
            return None
        sorted_returns = sorted(returns)
        cutoff = int(len(sorted_returns) * (Decimal("1") - confidence))
        tail = sorted_returns[:cutoff + 1]
        if not tail:
            return None
        return abs(sum(tail) / len(tail))

    @staticmethod
    def parametric_var(returns: list[Decimal], confidence: Decimal = Decimal("0.95")) -> Optional[Decimal]:
        """参数法 VaR（假设正态分布）。"""
        if len(returns) < 20:
            return None
        n = Decimal(len(returns))
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / n
        std = var.sqrt()
        # 95% → 1.645, 99% → 2.326
        z = Decimal("1.645") if confidence == Decimal("0.95") else Decimal("2.326")
        return abs(mean - z * std)

    @staticmethod
    def calculate_all(returns: list[Decimal], position_value: Decimal) -> dict:
        """一站式 VaR/CVaR 计算。"""
        hvar = VaRCalculator.historical_var(returns)
        cvar = VaRCalculator.historical_cvar(returns)
        return {"var_95": hvar, "cvar_95": cvar, "var_amount": hvar * position_value if hvar else None, "cvar_amount": cvar * position_value if cvar else None, "n_obs": len(returns)}
