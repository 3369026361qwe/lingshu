"""Copula 引擎 — 多资产尾部依赖建模 (v4.0).

纯计算层：无状态、无 IO。
替换线性 beta 缩放的压力测试，精确建模多资产联合尾部行为。

支持的 Copula:
    Clayton — 下尾依赖 (熊市齐跌)
    Gumbel  — 上尾依赖 (泡沫齐涨)
    t       — 对称厚尾
    Frank   — 无尾依赖 (基准)

数学基础:
    Sklar 定理: F(x1,...,xd) = C(F1(x1), ..., Fd(xd))

Usage:
    from jingsuan import CopulaEngine, CopulaType
    fit = CopulaEngine.fit(returns_matrix)
"""

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from shuju.utils import safe_divide

from ._copula_fit import (
    clayton_loglik,
    frank_loglik,
    gumbel_loglik,
    kendall_tau,
    pearson_correlation,
    t_copula_loglik,
    tcdf,
    theta_from_kendall,
)


class CopulaType(Enum):
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    T = "t"
    FRANK = "frank"


@dataclass
class CopulaFit:
    copula_type: CopulaType
    params: dict
    lower_tail_dep: Decimal
    upper_tail_dep: Decimal
    log_likelihood: float
    aic: float


class CopulaEngine:
    """Copula 连接函数引擎 — 多资产联合尾部建模。"""

    @staticmethod
    def fit(
        returns_matrix: list[list[Decimal]],
        types: list[CopulaType] | None = None,
    ) -> CopulaFit:
        """拟合所有 Copula 类型，返回 AIC 最优。"""
        if types is None:
            types = [CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.T, CopulaType.FRANK]

        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        if n_obs < 30:
            raise ValueError(f"Copula requires >=30 observations, got {n_obs}")

        pseudo_obs = CopulaEngine._to_pseudo_obs(returns_matrix, n_assets, n_obs)

        best_fit = None
        best_aic = float("inf")
        for ct in types:
            fit = None
            if ct == CopulaType.CLAYTON:
                fit = CopulaEngine._fit_clayton(pseudo_obs, n_obs)
            elif ct == CopulaType.GUMBEL:
                fit = CopulaEngine._fit_gumbel(pseudo_obs, n_obs)
            elif ct == CopulaType.T:
                fit = CopulaEngine._fit_t_copula(returns_matrix, n_assets, n_obs)
            elif ct == CopulaType.FRANK:
                fit = CopulaEngine._fit_frank(pseudo_obs, n_obs)
            if fit and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        return best_fit  # type: ignore[return-value]

    @staticmethod
    def _to_pseudo_obs(rmat, n_assets, n_obs):
        """转换为 pseudo-observations [0,1]."""
        result = []
        for a in range(n_assets):
            raw = rmat[a][:n_obs]
            sv = sorted(raw)
            result.append([(sv.index(v) + 1) / (n_obs + 1) for v in raw])
        return result

    # ── Fit wrappers ─────────────────────────────────────

    @staticmethod
    def _fit_clayton(po, n):
        a = max(0.1, min(theta_from_kendall(po, "clayton"), 20.0))
        ll = clayton_loglik(po, a, n)
        ld = Decimal(str(2 ** (-1 / a))) if a > 0 else Decimal("0")
        return CopulaFit(CopulaType.CLAYTON, {"theta": Decimal(str(round(a, 4)))},
                         ld, Decimal("0"), ll, 2.0 - 2.0 * ll)

    @staticmethod
    def _fit_gumbel(po, n):
        a = max(1.01, min(theta_from_kendall(po, "gumbel"), 20.0))
        ll = gumbel_loglik(po, a, n)
        ud = Decimal(str(2 - 2 ** (1 / a))) if a >= 1 else Decimal("0")
        return CopulaFit(CopulaType.GUMBEL, {"theta": Decimal(str(round(a, 4)))},
                         Decimal("0"), ud, ll, 2.0 - 2.0 * ll)

    @staticmethod
    def _fit_t_copula(rmat, n_assets, n_obs):
        rho_val = pearson_correlation(rmat[0][:n_obs], rmat[1][:n_obs])
        po = CopulaEngine._to_pseudo_obs(rmat, n_assets, n_obs)
        tau = kendall_tau(po[0], po[1])
        nu_val = max(2.5, 2 * tau / max(1 - tau, 0.01))
        ll = t_copula_loglik(rmat, rho_val, nu_val, n_assets, n_obs)
        arg = -math.sqrt((nu_val + 1) * (1 - abs(rho_val)) / (1 + abs(rho_val)))
        ld = Decimal(str(round(2 * tcdf(arg, nu_val + 1), 4)))
        return CopulaFit(CopulaType.T,
                         {"rho": Decimal(str(round(rho_val, 4))), "nu": Decimal(str(round(nu_val, 2)))},
                         ld, ld, ll, 4.0 - 2.0 * ll)

    @staticmethod
    def _fit_frank(po, n):
        tau = kendall_tau(po[0], po[1])
        a = max(-20.0, min(20.0, 9 * tau / max(0.01, 1 - tau * tau)))
        ll = frank_loglik(po, a, n)
        return CopulaFit(CopulaType.FRANK, {"theta": Decimal(str(round(a, 4)))},
                         Decimal("0"), Decimal("0"), ll, 2.0 - 2.0 * ll)

    # ── 模拟 ─────────────────────────────────────────────

    @staticmethod
    def simulate(fit, n_scenarios=10000):
        """从 Copula 采样生成联合情景 [0,1]^2."""
        import random
        random.seed(42)
        scenarios = []
        for _ in range(n_scenarios):
            u, v = random.random(), random.random()
            if fit.copula_type == CopulaType.CLAYTON:
                a = float(fit.params["theta"])
                v = (u ** (-a / (1 + a)) * (v ** (-a) - 1) + 1) ** (-1 / a)
            elif fit.copula_type == CopulaType.GUMBEL:
                a = float(fit.params["theta"])
                v = v ** (a / (a - 1)) if a > 1 else v
            elif fit.copula_type == CopulaType.T:
                r = float(fit.params["rho"])
                v = r * u + (1 - abs(r)) * v
            scenarios.append([u, v])
        return scenarios

    # ── 尾部依赖 ─────────────────────────────────────────

    @staticmethod
    def tail_dependence_matrix(rmat, method="nonparametric"):
        """两两资产尾部依赖系数矩阵."""
        n = len(rmat)
        result = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                ld = CopulaEngine._nonpar_lower_tail(rmat[i], rmat[j], Decimal("0.05"))
                result[i][j] = ld
                result[j][i] = ld
            result[i][i] = Decimal("1")
        return result

    @staticmethod
    def _nonpar_lower_tail(x, y, q):
        xq = sorted(x)[int(len(x) * float(q))]
        yq = sorted(y)[int(len(y) * float(q))]
        both = sum(1 for xi, yi in zip(x, y, strict=False) if xi <= xq and yi <= yq)
        xc = sum(1 for xi in x if xi <= xq)
        return safe_divide(both, xc)
