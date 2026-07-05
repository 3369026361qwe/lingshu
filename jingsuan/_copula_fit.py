"""Copula 拟合辅助 — 似然函数 + 统计工具 (v4.0 内部模块).

不直接导入。由 copula_engine.py 内部使用。
"""

import math
from decimal import Decimal

from shuju.utils import safe_mean


def kendall_tau(u: list[float], v: list[float]) -> float:
    """Kendall τ rank correlation."""
    n = len(u)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (u[i] - u[j]) * (v[i] - v[j]) > 0:
                c += 1
            elif (u[i] - u[j]) * (v[i] - v[j]) < 0:
                d += 1
    total = c + d
    return (c - d) / total if total > 0 else 0.0


def theta_from_kendall(po: list[list[float]], copula: str) -> float:
    """从 Kendall τ 反推 Copula 参数."""
    tau = kendall_tau(po[0], po[1])
    tau = max(-0.99, min(0.99, tau))
    if copula == "clayton":
        return 0.1 if tau <= 0 else 2 * tau / (1 - tau)
    elif copula == "gumbel":
        return 1.01 if tau <= 0 else 1 / max(0.01, 1 - tau)
    return 1.0


def pearson_correlation(x: list[Decimal], y: list[Decimal]) -> float:
    """Pearson 相关系数."""
    n = len(x)
    mx = float(safe_mean(x))
    my = float(safe_mean(y))
    sx = math.sqrt(sum((float(v) - mx) ** 2 for v in x) / max(1, n - 1))
    sy = math.sqrt(sum((float(v) - my) ** 2 for v in y) / max(1, n - 1))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((float(x[i]) - mx) * (float(y[i]) - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)


def clayton_loglik(po: list[list[float]], a: float, n: int) -> float:
    """Clayton log-likelihood."""
    ll = 0.0
    for t in range(n):
        u = max(1e-10, min(po[0][t], 1 - 1e-10))
        v = max(1e-10, min(po[1][t], 1 - 1e-10))
        try:
            term = u ** (-a) + v ** (-a) - 1
            if term <= 0:
                continue
            d = (1 + a) * (u * v) ** (-a - 1) * term ** (-1 / a - 2)
            if d > 0:
                ll += math.log(d)
        except (ValueError, OverflowError):
            continue
    return ll


def gumbel_loglik(po: list[list[float]], a: float, n: int) -> float:
    """Gumbel log-likelihood."""
    ll = 0.0
    for t in range(n):
        u = max(1e-10, min(po[0][t], 1 - 1e-10))
        v = max(1e-10, min(po[1][t], 1 - 1e-10))
        try:
            nlu = (-math.log(u)) ** a
            nlv = (-math.log(v)) ** a
            s = nlu + nlv
            cv = s ** (1 / a)
            d = ((cv + a - 1) / (u * v) * (nlu * nlv / s) ** (a - 1) * s ** (1 / a - 2))
            if d > 0:
                ll += math.log(d)
        except (ValueError, OverflowError):
            continue
    return ll


def t_copula_loglik(rmat: list[list[Decimal]], rho_val: float, nu_val: float,
                    n_assets: int, n_obs: int) -> float:
    """t-Copula log-likelihood (pairwise)."""
    ll = 0.0
    for t in range(n_obs):
        x = float(rmat[0][t])
        y = float(rmat[1][t])
        q = (x * x - 2 * rho_val * x * y + y * y) / (nu_val * (1 - rho_val * rho_val))
        d = (math.gamma((nu_val + 2) / 2)
             / (math.gamma(nu_val / 2) * nu_val * math.pi * math.sqrt(1 - rho_val * rho_val))
             * (1 + q) ** (-(nu_val + 2) / 2))
        if d > 0:
            ll += math.log(d)
    return ll


def tcdf(x: float, nu: float) -> float:
    """t distribution CDF (numerical integration)."""
    h = abs(x) / 100
    r = 0.0
    for i in range(100):
        ti = i * h
        pdf = (math.gamma((nu + 1) / 2)
               / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
               * (1 + ti * ti / nu) ** (-(nu + 1) / 2))
        r += pdf * h
    return 0.5 + r if x > 0 else 0.5 - r


def frank_loglik(po: list[list[float]], a: float, n: int) -> float:
    """Frank log-likelihood."""
    ll = 0.0
    et = math.exp(a)
    for t in range(n):
        u = max(1e-10, min(po[0][t], 1 - 1e-10))
        v = max(1e-10, min(po[1][t], 1 - 1e-10))
        try:
            num = a * (et - 1) * math.exp(a * (u + v))
            den = (et - math.exp(a * u) - math.exp(a * v) + math.exp(a * (u + v))) ** 2
            if den > 0 and num > 0:
                ll += math.log(num / den)
        except (ValueError, OverflowError):
            continue
    return ll
