"""
破产理论引擎 — Lundberg 不等式 + 动态风险预算 (v4.0).

纯计算层：无状态、无 IO。
将固定仓位限制 (10%/30%/95%) 替换为基于破产概率的动态风险预算。

数学基础:
    Cramér-Lundberg 模型离散近似:
        W_n = W_0 * Π(1 + f * r_i)
    破产概率: ψ(W_0, f) = P(min W_t < W_critical)
    Lundberg 不等式: ψ(u) <= exp(-R · u)
    最优仓位: f* = argmax E[W_T] subject to ψ <= ε

Usage:
    from jingsuan.ruin_engine import RuinEngine, RuinConfig
    f_star = RuinEngine.optimal_position_size(returns, config)
"""

import random
from dataclasses import dataclass
from decimal import Decimal
from math import exp, sqrt

from shuju.utils import safe_divide, safe_mean


@dataclass
class RuinConfig:
    initial_capital: Decimal = Decimal("1000000")
    ruin_threshold: Decimal = Decimal("500000")
    acceptable_ruin_prob: Decimal = Decimal("0.01")
    time_horizon: int = 252
    n_simulations: int = 10000   # 生产环境调整为 50000+
    random_seed: int = 42


@dataclass
class RuinAnalysis:
    ruin_probability: Decimal
    lundberg_bound: Decimal
    adjustment_coefficient: Decimal
    optimal_position_size: Decimal
    expected_terminal_wealth: Decimal
    expected_growth_rate: Decimal


class RuinEngine:
    """破产理论引擎 — Lundberg 模型 + 动态风险预算。"""

    @staticmethod
    def estimate_ruin_probability(
        trade_returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> Decimal:
        """蒙特卡洛 Bootstrap 估计破产概率.

        从历史交易回报中重采样，模拟 n_simulations 条路径，
        计算资产低于 ruin_threshold 的比例。
        """
        if config is None:
            config = RuinConfig()

        n_returns = len(trade_returns)
        if n_returns < 20:
            raise ValueError(f"需要 >= 20 笔历史交易, 当前 {n_returns}")

        rng = random.Random(config.random_seed)
        n_ruin = 0

        for _ in range(config.n_simulations):
            wealth = float(config.initial_capital)
            ruin_level = float(config.ruin_threshold)
            for _ in range(config.time_horizon):
                idx = rng.randint(0, n_returns - 1)
                r = float(trade_returns[idx])
                wealth *= (1 + float(position_size) * r)
                if wealth <= ruin_level:
                    n_ruin += 1
                    break

        return Decimal(n_ruin) / Decimal(config.n_simulations)

    @staticmethod
    def lundberg_bound(
        returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> Decimal:
        """Lundberg 指数上界: ψ(u) <= exp(-R * u).

        R > 0 是调节系数，满足 E[exp(R * Y - R * c * T)] = 1.
        在离散时间: R = 2 * (c - E[Y]) / Var[Y] (近似).
        """
        if config is None:
            config = RuinConfig()

        u = float(config.initial_capital - config.ruin_threshold)
        n = len(returns)
        # 单笔交易收益 = position_size * return
        trade_pnl = [float(position_size) * float(r) for r in returns]
        mu = sum(trade_pnl) / n
        var = sum((x - mu) ** 2 for x in trade_pnl) / max(1, n - 1)
        sigma = sqrt(var)

        if mu <= 0 or sigma == 0:
            return Decimal("1")  # 破产概率上界 = 100%

        # 正态近似下的调节系数
        R = 2 * mu / (sigma ** 2)
        if R <= 0:
            return Decimal("1")

        bound = exp(-R * u)
        return Decimal(str(min(bound, 1.0)))

    @staticmethod
    def optimal_position_size(
        trade_returns: list[Decimal],
        config: RuinConfig | None = None,
    ) -> RuinAnalysis:
        """二分搜索最优仓位: 最大化期望终值, ψ <= ε."""
        if config is None:
            config = RuinConfig()

        n = len(trade_returns)
        mu = float(safe_mean(trade_returns))
        var = sum((float(r) - mu) ** 2 for r in trade_returns) / max(1, n - 1)

        # Kelly 半仓作为上界 (更保守更安全)
        kelly_raw = safe_divide(mu, var) if var > 0 else Decimal("0.5")
        kelly_half = kelly_raw * Decimal("0.5")
        f_upper = max(min(kelly_half, Decimal("1")), Decimal("0.01"))
        f_lower = Decimal("0")

        best_f = Decimal("0")
        best_ew = Decimal("0")

        for _ in range(20):  # 二分
            f_mid = (f_lower + f_upper) / 2
            psi = RuinEngine.estimate_ruin_probability(trade_returns, f_mid, config)

            if psi <= config.acceptable_ruin_prob:
                ew = RuinEngine._expected_terminal(trade_returns, f_mid, config)
                if ew > best_ew:
                    best_ew = ew
                    best_f = f_mid
                f_lower = f_mid
            else:
                f_upper = f_mid

        rb = RuinEngine.estimate_ruin_probability(trade_returns, best_f, config)
        lb = RuinEngine.lundberg_bound(trade_returns, best_f, config)
        R_coef = Decimal(str(2 * mu / (var * float(best_f)) if var > 0 else 0))
        gr = RuinEngine._expected_growth(trade_returns, best_f)

        return RuinAnalysis(
            ruin_probability=rb,
            lundberg_bound=lb,
            adjustment_coefficient=R_coef,
            optimal_position_size=best_f,
            expected_terminal_wealth=best_ew,
            expected_growth_rate=gr,
        )

    @staticmethod
    def dynamic_risk_budget(
        current_drawdown: Decimal,
        max_position: Decimal,
    ) -> Decimal:
        """动态风险预算: 回撤越大, 仓位越小.

        实现 "survival first, profit second" 原则。
        budget = max_position * (1 - drawdown / 2)
        """
        shrink = Decimal("1") - safe_divide(current_drawdown, Decimal("2"))
        return max(Decimal("0.01"), max_position * max(shrink, Decimal("0")))

    # ── 内部方法 ─────────────────────────────────────────

    @staticmethod
    def _expected_terminal(
        returns: list[Decimal], f: Decimal, config: RuinConfig,
    ) -> Decimal:
        """估算期望终值 (解析近似: E[W_T] = W0 * (1 + f * μ)^T)."""
        mu = float(safe_mean(returns))
        wealth = float(config.initial_capital) * (1 + float(f) * mu) ** config.time_horizon
        return Decimal(str(max(wealth, float(config.ruin_threshold))))

    @staticmethod
    def _expected_growth(returns: list[Decimal], f: Decimal) -> Decimal:
        """期望增长率 g = (1/T) * E[ln(W_T/W_0)]."""
        mu = float(safe_mean(returns))
        var = sum((float(r) - mu) ** 2 for r in returns) / max(1, len(returns) - 1)
        # 近似: E[ln(1 + f * r)] ≈ f * μ - f^2 * var / 2
        g = float(f) * mu - float(f) ** 2 * var / 2
        return Decimal(str(max(g, 0.0)))
