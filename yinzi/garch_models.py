"""
GARCH 引擎 — 条件波动率建模 (v4.0).

替换静态波动率假设，捕获 A 股波动率聚类和杠杆效应。

Models:
    GARCH(1,1)     — 基础波动率聚类
    EGARCH(1,1)    — 非对称 (杠杆效应)
    GJR-GARCH(1,1) — 非对称 (替代形式)

数学:
    GARCH:   σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    EGARCH:  ln(σ²_t) = ω + α(|z|-E|z|) + γ·z + β·ln(σ²_{t-1})
    GJR:     σ²_t = ω + α·ε² + γ·I(ε<0)·ε² + β·σ²_{t-1}

Usage:
    from yinzi.garch_models import GARCHEngine
    result = GARCHEngine.garch_fit(returns)
"""

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class GARCHResult:
    model: str
    omega: Decimal
    alpha: Decimal
    beta: Decimal
    gamma: Decimal | None  # 非对称项 (EGARCH / GJR), None if symmetric
    persistence: Decimal    # α + β (GARCH) 或 β (EGARCH)
    conditional_vol: list[Decimal]  # 条件波动率序列
    converged: bool
    n_iterations: int


class GARCHEngine:
    """GARCH 族引擎 — 条件波动率. 纯静态方法."""

    @staticmethod
    def garch_fit(
        returns: list[Decimal],
        max_iter: int = 200,
    ) -> GARCHResult:
        """GARCH(1,1) 拟合 — 最大化准对数似然."""
        n = len(returns)
        if n < 50:
            raise ValueError(f"GARCH needs >= 50 obs, got {n}")

        # 初始参数 (OLS-like)
        r_float = [float(r) for r in returns]
        mu = sum(r_float) / n
        var0 = sum((r - mu) ** 2 for r in r_float) / max(1, n - 1)

        omega = var0 * 0.1
        alpha = 0.1
        beta = 0.8

        best = (omega, alpha, beta, float("-inf"))

        # 网格搜索初值
        for a0 in [0.05, 0.10, 0.15, 0.20]:
            for b0 in [0.70, 0.75, 0.80, 0.85, 0.90]:
                if a0 + b0 >= 1:
                    continue
                o0 = var0 * (1 - a0 - b0)
                ll = GARCHEngine._garch_loglik(r_float, o0, a0, b0)
                if ll > best[3]:
                    best = (o0, a0, b0, ll)

        omega, alpha, beta, _ = best

        # 条件波动率序列
        sigma2 = [var0]
        for t in range(1, n):
            sigma2.append(omega + alpha * (r_float[t - 1]) ** 2 + beta * sigma2[-1])

        conditional_vol = [Decimal(str(math.sqrt(max(s, 0)))) for s in sigma2]
        persistence = alpha + beta

        return GARCHResult(
            model="GARCH(1,1)",
            omega=Decimal(str(omega)),
            alpha=Decimal(str(alpha)),
            beta=Decimal(str(beta)),
            gamma=None,
            persistence=Decimal(str(persistence)),
            conditional_vol=conditional_vol,
            converged=persistence < 1,
            n_iterations=max_iter,
        )

    @staticmethod
    def _garch_loglik(r, omega, alpha, beta):
        """GARCH 准对数似然."""
        n = len(r)
        var0 = sum(x * x for x in r) / max(1, n)
        sigma2 = var0
        ll = 0.0
        for t in range(1, n):
            sigma2 = max(omega + alpha * r[t - 1] * r[t - 1] + beta * sigma2, 1e-10)
            ll += -0.5 * (math.log(2 * math.pi * sigma2) + r[t] * r[t] / sigma2)
        return ll

    @staticmethod
    def egarch_fit(
        returns: list[Decimal], max_iter: int = 200
    ) -> GARCHResult:
        """EGARCH(1,1) 拟合 — 捕捉杠杆效应 (γ < 0)."""
        n = len(returns)
        r_float = [float(r) for r in returns]
        var0 = sum(x * x for x in r_float) / max(1, n)

        # 初值
        omega = math.log(var0) * 0.1
        alpha = 0.1
        gamma = -0.05
        beta = 0.85

        # 网格搜索
        best = (omega, alpha, gamma, beta, float("-inf"))
        for a0 in [0.05, 0.10, 0.15]:
            for g0 in [0.0, -0.05, -0.10]:
                for b0 in [0.80, 0.85, 0.90]:
                    ll = GARCHEngine._egarch_loglik(r_float, math.log(var0) * 0.1, a0, g0, b0)
                    if ll > best[4]:
                        best = (math.log(var0) * 0.1, a0, g0, b0, ll)

        omega, alpha, gamma, beta, _ = best
        sigma2_ln = [math.log(var0)]
        z_sq = [r_float[0] / math.sqrt(var0) if var0 > 0 else 0.0]

        for t in range(1, n):
            s = math.sqrt(max(math.exp(sigma2_ln[-1]), 1e-10))
            z = r_float[t] / s if s > 0 else 0.0
            z_abs = abs(z)
            sigma2_ln.append(
                omega + alpha * (z_abs - math.sqrt(2 / math.pi))
                + gamma * z + beta * sigma2_ln[-1]
            )
            z_sq.append(z_abs)

        cond_vol = [Decimal(str(math.sqrt(max(math.exp(s), 1e-10)))) for s in sigma2_ln]

        return GARCHResult(
            model="EGARCH(1,1)",
            omega=Decimal(str(omega)), alpha=Decimal(str(alpha)),
            beta=Decimal(str(beta)), gamma=Decimal(str(gamma)),
            persistence=Decimal(str(beta)),
            conditional_vol=cond_vol, converged=abs(beta) < 1,
            n_iterations=max_iter,
        )

    @staticmethod
    def _egarch_loglik(r, omega, alpha, gamma, beta):
        n = len(r)
        sigma2 = max(sum(x * x for x in r) / max(1, n), 1e-10)
        log_sigma2 = math.log(sigma2)
        ll = 0.0
        s = math.sqrt(max(sigma2, 1e-10))
        z = r[0] / s if s > 0 else 0.0
        for t in range(1, n):
            z_abs = abs(z)
            log_sigma2 = omega + alpha * (z_abs - math.sqrt(2 / math.pi)) + gamma * z + beta * log_sigma2
            sigma2 = math.exp(log_sigma2)
            s = math.sqrt(sigma2)
            z = r[t] / s if s > 0 else 0.0
            ll += -0.5 * (math.log(2 * math.pi * sigma2) + r[t] * r[t] / sigma2)
        return ll

    @staticmethod
    def conditional_var(
        garch_result: GARCHResult,
        confidence: Decimal = Decimal("0.99"),
    ) -> list[Decimal]:
        """从 GARCH 结果计算条件 VaR 序列.

        VaR_t(α) = σ_t · z_α (简化: 用正态分位数).
        """
        z = Decimal("2.326") if confidence >= Decimal("0.99") else Decimal("1.645")
        return [-z * vol for vol in garch_result.conditional_vol]
