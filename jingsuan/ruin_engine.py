"""
破产理论引擎 — Cramér-Lundberg + Beekman-Bowers + 动态风险预算 (v4.1).

纯计算层：无状态、无 IO。
将固定仓位限制 (10%/30%/95%) 替换为基于破产概率的动态风险预算。

v4.1 增强:
    - Cramér-Lundberg 精确解 (compound Poisson process)
    - Beekman-Bowers 近似
    - 多期破产概率的条件更新 (Bayesian update)

数学基础:
    Cramér-Lundberg 模型:
        R(t) = u + c·t - Σ_{i=1}^{N(t)} Y_i
    Lundberg 不等式: ψ(u) <= exp(-R·u)
    调节系数 R: E[exp(R·Y - R·c·T)] = 1

    Beekman-Bowers 近似:
        ψ(u) ≈ (1 - ρ) · (1 - F_e(u))
        where ρ = E[Y]/(c·E[T]), F_e = equilibrium distribution of Y

Usage:
    from jingsuan.ruin_engine import RuinEngine, RuinConfig
    f_star = RuinEngine.optimal_position_size(returns, config)
"""

import math
import random
from dataclasses import dataclass, field
from decimal import Decimal

from shuju.utils import safe_divide, safe_mean

# ── Dataclasses ───────────────────────────────────────────

@dataclass
class RuinConfig:
    initial_capital: Decimal = Decimal("1000000")
    ruin_threshold: Decimal = Decimal("500000")
    acceptable_ruin_prob: Decimal = Decimal("0.01")
    time_horizon: int = 252
    n_simulations: int = 10000   # 生产环境调整为 50000+
    random_seed: int = 42
    premium_rate: Decimal = Decimal("0")   # 保费率 c (交易中 = 预期收益)


@dataclass
class RuinAnalysis:
    ruin_probability: Decimal
    lundberg_bound: Decimal
    adjustment_coefficient: Decimal
    optimal_position_size: Decimal
    expected_terminal_wealth: Decimal
    expected_growth_rate: Decimal
    # v4.1 additions
    beekman_bowers_approx: Decimal = Decimal("0")
    cramer_lundberg_exact: Decimal = Decimal("0")
    conditional_survival: list[Decimal] = field(default_factory=list)


@dataclass
class MultiPeriodRuin:
    """多期破产概率 (条件更新)."""
    periods: list[int]                   # [1, 5, 21, 63, 126, 252]
    ruin_probs: list[Decimal]            # 每期的无条件破产概率
    conditional_ruin_probs: list[Decimal]  # 给定前期未破产的条件概率
    survival_curve: list[Decimal]        # 生存概率 S(t) = P(no ruin by t)


class RuinEngine:
    """破产理论引擎 — Cramér-Lundberg + Beekman-Bowers + 动态风险预算。"""

    # ═══════════════════════════════════════════════════════
    # Monte Carlo Ruin Probability
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def estimate_ruin_probability(
        trade_returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> Decimal:
        """蒙特卡洛 Bootstrap 估计破产概率."""
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

    # ═══════════════════════════════════════════════════════
    # Multi-period Ruin (Conditional Updates)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def multi_period_ruin(
        trade_returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> MultiPeriodRuin:
        """多期破产概率及条件更新.

        Computes ψ(t) = P(ruin occurs by day t) for t ∈ periods.
        Then: ψ(t|survived to t-1) = ψ(t) - ψ(t-1) / (1 - ψ(t-1))
        """
        if config is None:
            config = RuinConfig()

        n_returns = len(trade_returns)
        rng = random.Random(config.random_seed)

        # Default periods: daily, weekly, monthly, quarterly, semi-annual, annual
        periods = [1, 5, 21, 63, 126, 252]
        # Cap at time_horizon
        periods = [p for p in periods if p <= config.time_horizon]

        # Track ruin at each cutoff in one simulation pass
        ruin_cum = [0] * len(periods)

        for _ in range(config.n_simulations):
            wealth = float(config.initial_capital)
            ruin_level = float(config.ruin_threshold)
            ruined = False
            for t in range(config.time_horizon):
                idx = rng.randint(0, n_returns - 1)
                r = float(trade_returns[idx])
                wealth *= (1 + float(position_size) * r)
                if wealth <= ruin_level and not ruined:
                    ruined = True
                    # Record ruin at each period >= current t
                    for pi, p in enumerate(periods):
                        if t < p:
                            ruin_cum[pi] += 1
                    break

        n_sims = config.n_simulations
        ruin_probs = [Decimal(rc) / Decimal(n_sims) for rc in ruin_cum]

        # Conditional ruin: ψ(t_i | survived to t_{i-1}) = (ψ(t_i) - ψ(t_{i-1})) / (1 - ψ(t_{i-1}))
        cond_ruin = [ruin_probs[0]]
        for i in range(1, len(ruin_probs)):
            prev_survival = Decimal("1") - ruin_probs[i - 1]
            if prev_survival > 0:
                delta = ruin_probs[i] - ruin_probs[i - 1]
                cond_ruin.append(safe_divide(delta, prev_survival))
            else:
                cond_ruin.append(Decimal("1"))

        # Survival curve
        survival = [Decimal("1") - rp for rp in ruin_probs]

        return MultiPeriodRuin(
            periods=periods,
            ruin_probs=ruin_probs,
            conditional_ruin_probs=cond_ruin,
            survival_curve=survival,
        )

    @staticmethod
    def update_ruin_belief(
        prior_ruin: MultiPeriodRuin,
        survived_days: int,
    ) -> MultiPeriodRuin:
        """Bayesian update of ruin probabilities given survival.

        Given the portfolio has survived for `survived_days`, update
        the ruin probabilities using Bayes rule.

        New ruin probability at horizon T:
            ψ_new(T) = (ψ(T) - ψ(survived)) / (1 - ψ(survived))
        """
        # Find the closest period index
        idx = 0
        for i, p in enumerate(prior_ruin.periods):
            if p <= survived_days:
                idx = i
            else:
                break

        survivor_prob = Decimal("1") - prior_ruin.ruin_probs[idx]
        if survivor_prob <= 0:
            return prior_ruin  # Already ruined

        new_ruin_probs = []
        for i, p in enumerate(prior_ruin.periods):
            if p <= survived_days:
                new_ruin_probs.append(Decimal("0"))  # Already survived, probability 0
            else:
                new_rp = safe_divide(
                    prior_ruin.ruin_probs[i] - prior_ruin.ruin_probs[idx],
                    survivor_prob,
                )
                new_ruin_probs.append(new_rp)

        new_cond = [new_ruin_probs[0]]
        for i in range(1, len(new_ruin_probs)):
            prev_surv = Decimal("1") - new_ruin_probs[i - 1]
            if prev_surv > 0:
                delta = new_ruin_probs[i] - new_ruin_probs[i - 1]
                new_cond.append(safe_divide(delta, prev_surv))
            else:
                new_cond.append(Decimal("1"))

        return MultiPeriodRuin(
            periods=prior_ruin.periods,
            ruin_probs=new_ruin_probs,
            conditional_ruin_probs=new_cond,
            survival_curve=[Decimal("1") - rp for rp in new_ruin_probs],
        )

    # ═══════════════════════════════════════════════════════
    # Cramér-Lundberg Exact Solution
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def cramer_lundberg_exact(
        trade_returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> Decimal:
        """Cramér-Lundberg 精确解 (compound Poisson with exponential claims).

        假设:
            - 盈利事件服从 Poisson(λ) 过程
            - 单次亏损额服从 Exp(1/μ_Y) 分布
            - 保费率 c = 预期收益 / 时间

        精确破产概率:
            ψ(u) = (1/(1+θ)) · exp(-θ·ρ·u / (μ_Y·(1+θ)))

        其中:
            θ = c / (λ·μ_Y) - 1  (安全负荷)
            ρ = λ·μ_Y / c         (调整后负荷率)
            u = 初始盈余

        Args:
            trade_returns: 单笔交易收益序列
            position_size: 仓位大小
            config: 破产配置

        Returns:
            精确破产概率 (在复合Poisson+指数索赔假设下)
        """
        if config is None:
            config = RuinConfig()

        u = float(config.initial_capital - config.ruin_threshold)
        pass  # (n.b.: length checked upstream)
        # Separate gains and losses
        trade_pnl = [float(position_size) * float(r) for r in trade_returns]
        losses = [-x for x in trade_pnl if x < 0]
        gains = [x for x in trade_pnl if x > 0]

        if not losses:
            return Decimal("0")

        # Rate parameters
        lam = len(losses) / len(trade_returns)  # loss frequency
        mu_Y = sum(losses) / len(losses) if losses else 0.0  # mean loss size
        c = sum(gains) / len(trade_returns) if gains else 0.0  # premium rate

        if mu_Y <= 0 or c <= 0 or lam <= 0:
            return Decimal("1")

        # Safety loading
        theta = c / (lam * mu_Y) - 1.0 if lam * mu_Y > 0 else 0.0

        if theta <= 0:
            # Negative safety loading → ruin is certain
            return Decimal("1")

        # Exact ruin probability
        psi = (1.0 / (1.0 + theta)) * math.exp(-theta * u / (mu_Y * (1.0 + theta)))
        psi = max(0.0, min(psi, 1.0))

        return Decimal(str(round(psi, 8)))

    # ═══════════════════════════════════════════════════════
    # Beekman-Bowers Approximation
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def beekman_bowers_approx(
        trade_returns: list[Decimal],
        position_size: Decimal,
        config: RuinConfig | None = None,
    ) -> Decimal:
        """Beekman-Bowers 近似.

        ψ_BB(u) ≈ (1 - ρ) · (1 - F_e(u))

        其中:
            ρ = E[Y] / (c · E[T])  — 损失负荷率
            F_e(u) = (1/E[Y]) · ∫_0^u (1 - F_Y(y)) dy  — 损失额的均衡分布 CDF

        对任意损失分布成立 (不需要指数假设)。
        """
        if config is None:
            config = RuinConfig()

        u = float(config.initial_capital - config.ruin_threshold)
        pass  # (n.b.: length checked upstream)
        # PnL
        trade_pnl = [float(position_size) * float(r) for r in trade_returns]
        losses = [-x for x in trade_pnl if x < 0]
        gains = [x for x in trade_pnl if x > 0]

        if not losses:
            return Decimal("0")

        # Loss distribution moments
        mu_Y = sum(losses) / len(losses) if losses else 0.0
        sum(y * y for y in losses) / len(losses) if losses else 0.0

        # Premium rate c
        c = sum(gains) / len(trade_returns) if gains else mu_Y * 1.5  # default safety loading

        if mu_Y <= 0 or c <= 0:
            return Decimal("1")

        # Loss load ρ = E[Y] / c
        rho = mu_Y / c
        if rho >= 1:
            return Decimal("1")

        # Equilibrium distribution F_e(u):
        # For exponential claims: F_e(u) = 1 - exp(-u / μ_Y)
        # General: use empirical CDF integration
        losses_sorted = sorted(losses)

        # Compute F_e(u) via numerical integration
        # F_e(u) = (1/μ_Y) · Σ (u - y_i) · I(y_i < u) / n_losses (discrete approx)
        # Better: use the empirical tail integral
        fe_u = 0.0
        for y in losses_sorted:
            if y < u:
                fe_u += (u - y)
        fe_u = fe_u / (mu_Y * len(losses)) if len(losses) > 0 else 0.0
        fe_u = min(fe_u, 1.0)

        # Beekman-Bowers formula
        psi_bb = (1.0 - rho) * (1.0 - fe_u)
        psi_bb = max(0.0, min(psi_bb, 1.0))

        return Decimal(str(round(psi_bb, 8)))

    # ═══════════════════════════════════════════════════════
    # Lundberg Bound
    # ═══════════════════════════════════════════════════════

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
        trade_pnl = [float(position_size) * float(r) for r in returns]
        mu = sum(trade_pnl) / n
        var = sum((x - mu) ** 2 for x in trade_pnl) / max(1, n - 1)
        sigma = math.sqrt(var)

        if mu <= 0 or sigma == 0:
            return Decimal("1")

        # 正态近似下的调节系数
        R = 2 * mu / (sigma ** 2)
        if R <= 0:
            return Decimal("1")

        bound = math.exp(-R * u)
        return Decimal(str(min(bound, 1.0)))

    # ═══════════════════════════════════════════════════════
    # Optimal Position Size
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def optimal_position_size(
        trade_returns: list[Decimal],
        config: RuinConfig | None = None,
    ) -> RuinAnalysis:
        """二分搜索最优仓位: 最大化期望终值, ψ <= ε.

        Uses both Cramér-Lundberg exact + Monte Carlo verification.
        """
        if config is None:
            config = RuinConfig()

        n = len(trade_returns)
        mu = float(safe_mean(trade_returns))
        var = sum((float(r) - mu) ** 2 for r in trade_returns) / max(1, n - 1)

        # Kelly 半仓作为上界
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

        # Consolidated analysis
        rb = RuinEngine.estimate_ruin_probability(trade_returns, best_f, config)
        lb = RuinEngine.lundberg_bound(trade_returns, best_f, config)
        R_coef = Decimal(str(2 * mu / (var * float(best_f)) if var > 0 else 0))
        gr = RuinEngine._expected_growth(trade_returns, best_f)
        bb = RuinEngine.beekman_bowers_approx(trade_returns, best_f, config)
        cl = RuinEngine.cramer_lundberg_exact(trade_returns, best_f, config)

        # Multi-period analysis
        mp = RuinEngine.multi_period_ruin(trade_returns, best_f, config)

        return RuinAnalysis(
            ruin_probability=rb,
            lundberg_bound=lb,
            adjustment_coefficient=R_coef,
            optimal_position_size=best_f,
            expected_terminal_wealth=best_ew,
            expected_growth_rate=gr,
            beekman_bowers_approx=bb,
            cramer_lundberg_exact=cl,
            conditional_survival=mp.survival_curve,
        )

    # ═══════════════════════════════════════════════════════
    # Dynamic Risk Budget
    # ═══════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════

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
        g = float(f) * mu - float(f) ** 2 * var / 2
        return Decimal(str(max(g, 0.0)))
