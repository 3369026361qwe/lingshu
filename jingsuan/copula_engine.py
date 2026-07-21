"""
Copula 引擎 — 多资产尾部依赖建模 (v4.2).

纯计算层：无状态、无 IO。
替换线性 beta 缩放的压力测试，精确建模多资产联合尾部行为。

支持的 Copula:
    Clayton       — 下尾依赖 (熊市齐跌)
    Gumbel        — 上尾依赖 (泡沫齐涨)
    RotatedGumbel — 下尾依赖 (Survival Gumbel, 180° rotation)
    t             — 对称厚尾
    Frank         — 无尾依赖 (基准)
    Gaussian      — 无尾依赖 (基准, 线性相关)

Usage:
    from jingsuan.copula_engine import CopulaEngine, CopulaType
    fit = CopulaEngine.fit(returns_matrix)
"""

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from jingsuan._copula_fit import (
    clayton_loglik,
    frank_loglik,
    gumbel_loglik,
    kendall_tau,
    pearson_correlation,
    t_copula_loglik,
    tcdf,
    theta_from_kendall,
)
from shuju.utils import safe_divide

# -- Shared math utilities (also usable by other modules) --

def _std_norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf. Accurate to ~1e-15."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


_GL_NODES = [
    -0.9815606342467192, -0.9041172563704749, -0.7699026741943047,
    -0.5873179542866175, -0.3678314989981802, -0.1252334085114689,
    0.1252334085114689, 0.3678314989981802, 0.5873179542866175,
    0.7699026741943047, 0.9041172563704749, 0.9815606342467192,
]
_GL_WEIGHTS = [
    0.04717533638651183, 0.10693932599531843, 0.16007832854334622,
    0.20316742672306592, 0.23349253653835480, 0.24914704581340277,
    0.24914704581340277, 0.23349253653835480, 0.20316742672306592,
    0.16007832854334622, 0.10693932599531843, 0.04717533638651183,
]


# -- Enum --

class CopulaType(Enum):
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    ROTATED_GUMBEL = "rotated_gumbel"
    T = "t"
    FRANK = "frank"
    GAUSSIAN = "gaussian"


# -- Dataclasses --

@dataclass
class CopulaFit:
    copula_type: CopulaType
    params: dict
    lower_tail_dep: Decimal
    upper_tail_dep: Decimal
    log_likelihood: float
    aic: float
    cvm_statistic: float = 0.0
    cvm_pvalue: float = 1.0
    n_dimensions: int = 2


@dataclass
class MultiCopulaFit:
    copula_type: CopulaType
    params: dict
    correlation_matrix: list[list[Decimal]]
    lower_tail_dep_matrix: list[list[Decimal]]
    upper_tail_dep_matrix: list[list[Decimal]]
    log_likelihood: float
    aic: float
    n_dimensions: int
    cvm_statistic: float = 0.0
    cvm_pvalue: float = 1.0


# -- Copula Engine --

class CopulaEngine:
    """Copula 连接函数引擎 — 多资产联合尾部建模。"""

    # ============================================================
    # Fit (main entry point)
    # ============================================================

    @staticmethod
    def fit(
        returns_matrix: list[list[Decimal]],
        types: list[CopulaType] | None = None,
        multi_dimensional: bool = False,
        n_bootstrap_gof: int = 200,
    ) -> CopulaFit:
        """拟合所有 Copula 类型，返回 AIC 最优。"""
        if types is None:
            types = [
                CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.ROTATED_GUMBEL,
                CopulaType.T, CopulaType.FRANK, CopulaType.GAUSSIAN,
            ]
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        if n_obs < 30:
            raise ValueError(f"Copula requires >=30 observations, got {n_obs}")

        if multi_dimensional and n_assets > 2:
            return CopulaEngine._fit_multidimensional(
                returns_matrix, n_assets, n_obs, types, n_bootstrap_gof
            )

        pseudo_obs = CopulaEngine._to_pseudo_obs(returns_matrix, n_assets, n_obs)
        best_fit: CopulaFit | None = None
        best_aic = float("inf")
        for ct in types:
            fit = CopulaEngine._fit_single(ct, pseudo_obs, returns_matrix, n_assets, n_obs)
            if fit and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
        if best_fit is None:
            raise RuntimeError("All Copula fits failed")
        return CopulaEngine._gof_cramer_von_mises(best_fit, pseudo_obs, n_obs, n_bootstrap_gof)

    @staticmethod
    def _fit_single(ct, pseudo_obs, returns_matrix, n_assets, n_obs) -> CopulaFit | None:
        if ct == CopulaType.CLAYTON:
            return CopulaEngine._fit_clayton(pseudo_obs, n_obs)
        elif ct == CopulaType.GUMBEL:
            return CopulaEngine._fit_gumbel(pseudo_obs, n_obs)
        elif ct == CopulaType.ROTATED_GUMBEL:
            return CopulaEngine._fit_rotated_gumbel(pseudo_obs, n_obs)
        elif ct == CopulaType.T:
            return CopulaEngine._fit_t_copula(returns_matrix, n_assets, n_obs)
        elif ct == CopulaType.FRANK:
            return CopulaEngine._fit_frank(pseudo_obs, n_obs)
        elif ct == CopulaType.GAUSSIAN:
            return CopulaEngine._fit_gaussian(pseudo_obs, n_obs)
        return None

    # ============================================================
    # Pseudo observations
    # ============================================================

    @staticmethod
    def _to_pseudo_obs(rmat, n_assets, n_obs):
        """Convert to pseudo-observations in [0,1] via empirical CDF."""
        result = []
        for a in range(n_assets):
            raw = rmat[a][:n_obs]
            sv = sorted(raw)
            result.append([(sv.index(v) + 1) / (n_obs + 1) for v in raw])
        return result

    # ============================================================
    # Individual Copula fits
    # ============================================================

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
    def _fit_rotated_gumbel(po, n):
        """Rotated Gumbel (180°) = Survival Gumbel. Captures lower-tail dependence."""
        po_rotated = [[1.0 - u for u in po[0]], [1.0 - v for v in po[1]]]
        a = max(1.01, min(theta_from_kendall(po_rotated, "gumbel"), 20.0))
        ll = gumbel_loglik(po_rotated, a, n)
        ld = Decimal(str(2 - 2 ** (1 / a))) if a >= 1 else Decimal("0")
        return CopulaFit(CopulaType.ROTATED_GUMBEL, {"theta": Decimal(str(round(a, 4)))},
                         ld, Decimal("0"), ll, 2.0 - 2.0 * ll)

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

    @staticmethod
    def _fit_gaussian(po, n):
        """Gaussian Copula via Kendall-tau inversion: rho = sin(pi*tau/2)."""
        tau = kendall_tau(po[0], po[1])
        rho = math.sin(math.pi * tau / 2)
        rho = max(-0.999, min(rho, 0.999))
        ll = 0.0
        for t in range(n):
            u = max(1e-10, min(po[0][t], 1 - 1e-10))
            v = max(1e-10, min(po[1][t], 1 - 1e-10))
            try:
                x, y = CopulaEngine._norm_inv(u), CopulaEngine._norm_inv(v)
                q = (x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho))
                d = 1.0 / (2 * math.pi * math.sqrt(1 - rho * rho)) * math.exp(-q)
                if d > 0:
                    ll += math.log(d)
            except (ValueError, OverflowError):
                continue
        return CopulaFit(CopulaType.GAUSSIAN, {"rho": Decimal(str(round(rho, 4)))},
                         Decimal("0"), Decimal("0"), ll, 2.0 - 2.0 * ll)

    @staticmethod
    def _norm_inv(p: float) -> float:
        """Inverse normal CDF (Abramowitz & Stegun rational approximation)."""
        if p <= 0:
            return -8.0
        if p >= 1:
            return 8.0
        q = p - 0.5
        if abs(q) <= 0.425:
            r_val = 0.180625 - q * q
            num = ((((((2.5090809287301226727e3 * r_val + 3.3430575583588128105e4) * r_val
                       + 6.7265770927008700853e4) * r_val + 4.5921953931549871457e4) * r_val
                     + 1.3731693765509461125e4) * r_val + 1.9715909503065514427e3) * r_val
                   + 1.3314166789178437745e2) * r_val + 3.3871328727963666080e0
            den = ((((((5.2264952788528545610e3 * r_val + 2.8729085735721942674e4) * r_val
                      + 3.9307895800092710610e4) * r_val + 2.1213794301586595867e4) * r_val
                    + 5.3941960214247511077e3) * r_val + 6.8718700749205790830e2) * r_val
                  + 4.2313330701600911252e1) * r_val + 1.0
            return q * num / den
        else:
            r_val = p if q > 0 else 1 - p
            r_val = math.sqrt(-math.log(r_val))
            x_num = (((((2.0103219639223422879e-2 * r_val + 5.3203017571162469336e-2) * r_val
                        + 1.3427395851606236084e-1) * r_val
                      + 2.3003768220797445470e-1) * r_val + 2.8551022315479562992e-1) * r_val
                    + 3.0490228038030902545e-1) * r_val + 1.0
            return -x_num if q < 0 else x_num

    # ============================================================
    # Multi-dimensional (Vine Copula decomposition)
    # ============================================================

    @staticmethod
    def _fit_multidimensional(returns_matrix, n_assets, n_obs, types, n_bootstrap):
        """Multi-dimensional via pairwise Copula fits + correlation matrix."""
        po = CopulaEngine._to_pseudo_obs(returns_matrix, n_assets, n_obs)
        lt_matrix = [[Decimal("0")] * n_assets for _ in range(n_assets)]
        ut_matrix = [[Decimal("0")] * n_assets for _ in range(n_assets)]
        corr_matrix = [[Decimal("0")] * n_assets for _ in range(n_assets)]
        pairwise_params = {}
        total_ll = 0.0

        for i in range(n_assets):
            corr_matrix[i][i] = Decimal("1")
            for j in range(i + 1, n_assets):
                pair_po = [po[i], po[j]]
                pair_rmat = [returns_matrix[i], returns_matrix[j]]
                fit_ij = CopulaEngine._fit_best_pair(pair_po, pair_rmat, 2, n_obs, types)
                if fit_ij:
                    lt_matrix[i][j] = lt_matrix[j][i] = fit_ij.lower_tail_dep
                    ut_matrix[i][j] = ut_matrix[j][i] = fit_ij.upper_tail_dep
                    corr_ij = Decimal(str(pearson_correlation(
                        returns_matrix[i][:n_obs], returns_matrix[j][:n_obs])))
                    corr_matrix[i][j] = corr_matrix[j][i] = corr_ij
                    total_ll += fit_ij.log_likelihood
                    pairwise_params[f"{i},{j}"] = {
                        "copula_type": fit_ij.copula_type.value,
                        "params": {k: str(v) for k, v in fit_ij.params.items()},
                        "lower_tail_dep": str(fit_ij.lower_tail_dep),
                        "upper_tail_dep": str(fit_ij.upper_tail_dep),
                    }

        n_params = sum(1 + len(f["params"]) for f in pairwise_params.values())
        return MultiCopulaFit(
            copula_type=CopulaType.T,
            params={"pairwise": pairwise_params},
            correlation_matrix=corr_matrix,
            lower_tail_dep_matrix=lt_matrix,
            upper_tail_dep_matrix=ut_matrix,
            log_likelihood=total_ll,
            aic=2 * n_params - 2 * total_ll,
            n_dimensions=n_assets,
        )

    @staticmethod
    def _fit_best_pair(pseudo_obs, rmat, n_assets, n_obs, types):
        best_fit, best_aic = None, float("inf")
        for ct in types:
            fit = CopulaEngine._fit_single(ct, pseudo_obs, rmat, n_assets, n_obs)
            if fit and fit.aic < best_aic:
                best_aic, best_fit = fit.aic, fit
        return best_fit

    # ============================================================
    # Cramér-von Mises Goodness-of-Fit
    # ============================================================

    @staticmethod
    def _gof_cramer_von_mises(fit, pseudo_obs, n, n_bootstrap=200):
        """Cramér-von Mises GoF with parametric bootstrap p-value."""
        if n < 20:
            return fit
        cvm_obs = CopulaEngine._compute_cvm(pseudo_obs, fit, n)
        rng = random.Random(42)
        cvm_boot = []
        for _ in range(n_bootstrap):
            sim_data_raw = CopulaEngine.simulate(fit, n, seed=rng.randint(1, 999999))
            sim_data = [[s[0] for s in sim_data_raw], [s[1] for s in sim_data_raw]]
            sv0, sv1 = sorted(sim_data[0]), sorted(sim_data[1])
            boot_po = [
                [(sv0.index(v) + 1) / (n + 1) for v in sim_data[0]],
                [(sv1.index(v) + 1) / (n + 1) for v in sim_data[1]],
            ]
            cvm_boot.append(CopulaEngine._compute_cvm(boot_po, fit, n))
        n_extreme = sum(1 for c in cvm_boot if c >= cvm_obs)
        fit.cvm_statistic = round(cvm_obs, 6)
        fit.cvm_pvalue = round((n_extreme + 1) / (n_bootstrap + 1), 4)
        return fit

    @staticmethod
    def _compute_cvm(pseudo_obs, fit, n):
        cvm = 0.0
        for t in range(n):
            u, v = pseudo_obs[0][t], pseudo_obs[1][t]
            c_emp = sum(1 for s in range(n)
                        if pseudo_obs[0][s] <= u and pseudo_obs[1][s] <= v) / n
            c_theta = CopulaEngine._copula_cdf(fit, u, v)
            cvm += (c_emp - c_theta) ** 2
        return cvm

    @staticmethod
    def _copula_cdf(fit, u, v):
        """Analytic Copula CDF for each supported type."""
        ct = fit.copula_type
        try:
            if ct == CopulaType.CLAYTON:
                theta = float(fit.params["theta"])
                if theta <= 0:
                    return u * v
                return max(0.0, (u ** (-theta) + v ** (-theta) - 1) ** (-1.0 / theta))
            elif ct == CopulaType.GUMBEL:
                theta = float(fit.params["theta"])
                if theta < 1:
                    return u * v
                return math.exp(-((-math.log(max(u, 1e-10))) ** theta
                                  + (-math.log(max(v, 1e-10))) ** theta) ** (1.0 / theta))
            elif ct == CopulaType.ROTATED_GUMBEL:
                theta = float(fit.params["theta"])
                if theta < 1:
                    return u * v
                ur, vr = 1 - u, 1 - v
                c_gumbel = math.exp(-((-math.log(max(ur, 1e-10))) ** theta
                                      + (-math.log(max(vr, 1e-10))) ** theta) ** (1.0 / theta))
                return max(0.0, u + v - 1 + c_gumbel)
            elif ct == CopulaType.FRANK:
                theta = float(fit.params["theta"])
                if abs(theta) < 1e-10:
                    return u * v
                e = math.exp(theta)
                num = (math.exp(theta * u) - 1) * (math.exp(theta * v) - 1)
                return max(0.0, math.log(1 + num / (e - 1)) / theta)
            elif ct in (CopulaType.T, CopulaType.GAUSSIAN):
                rho = float(fit.params["rho"])
                x, y = CopulaEngine._norm_inv(u), CopulaEngine._norm_inv(v)
                return _bivariate_normal_cdf(x, y, rho)
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        return u * v

    # ============================================================
    # Conditional sampling (h-function inversion)
    # ============================================================

    @staticmethod
    def conditional_sampling(fit, condition, n_scenarios=10000, seed=42):
        """Conditional quantile sampling. Returns [{index: quantile}, ...]."""
        rng = random.Random(seed)
        scenarios = []
        for _ in range(n_scenarios):
            scenario = dict(condition)
            w = rng.random()
            if fit.n_dimensions == 2 and len(condition) == 1:
                u = list(condition.values())[0]
                scenario[1 - list(condition.keys())[0]] = CopulaEngine._h_inverse(fit, u, w)
            else:
                for k in range(fit.n_dimensions):
                    if k not in condition:
                        scenario[k] = rng.random()
            scenarios.append(scenario)
        return scenarios

    @staticmethod
    def _h_inverse(fit, u, w):
        """Bisection inversion of h(v|u)=w. Returns v."""
        lo, hi = 0.001, 0.999
        for _ in range(30):
            mid = (lo + hi) / 2
            if CopulaEngine._h_function(fit, u, mid) < w:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def _h_function(fit, u, v):
        """h(v|u) = dC(u,v)/du (conditional distribution)."""
        ct = fit.copula_type
        try:
            if ct == CopulaType.CLAYTON:
                theta = float(fit.params["theta"])
                s = u ** (-theta) + v ** (-theta) - 1
                if s <= 0:
                    return v
                return (u ** (-theta - 1)) * s ** (-1.0 / theta - 1)
            elif ct == CopulaType.GUMBEL:
                theta = float(fit.params["theta"])
                nlu = (-math.log(max(u, 1e-15))) ** theta
                nlv = (-math.log(max(v, 1e-15))) ** theta
                s = nlu + nlv
                if s <= 0:
                    return v
                cv = s ** (1.0 / theta)
                return cv * nlu / (u * (-math.log(max(u, 1e-15))) * s) * math.exp(-cv)
            elif ct == CopulaType.ROTATED_GUMBEL:
                return 1.0 - CopulaEngine._h_function(
                    CopulaFit(CopulaType.GUMBEL, fit.params, Decimal("0"), Decimal("0"), 0.0, 0.0),
                    1 - u, 1 - v)
            elif ct in (CopulaType.T, CopulaType.GAUSSIAN):
                rho = float(fit.params["rho"])
                x, y = CopulaEngine._norm_inv(u), CopulaEngine._norm_inv(v)
                return _std_norm_cdf((y - rho * x) / math.sqrt(max(1 - rho * rho, 1e-15)))
            else:
                return v
        except (ValueError, OverflowError, ZeroDivisionError):
            return v

    # ============================================================
    # Simulation
    # ============================================================

    @staticmethod
    def simulate(fit, n_scenarios=10000, seed=None):
        """Generate joint uniform samples.

        Uses exact conditional sampling for Clayton/Gumbel/Frank;
        Cholesky-based method for Gaussian/t Copula.
        Returns [[u1, v1], [u2, v2], ...] (scenario-major).
        """
        rng = random.Random(seed)
        ct = fit.copula_type
        scenarios = []

        if ct in (CopulaType.GAUSSIAN, CopulaType.T):
            rho = float(fit.params["rho"])
            chol = [[1.0, 0.0], [rho, math.sqrt(max(1 - rho * rho, 1e-15))]]
            if ct == CopulaType.T:
                nu = float(fit.params["nu"])
                for _ in range(n_scenarios):
                    z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
                    x = chol[0][0] * z1 + chol[0][1] * z2
                    y = chol[1][0] * z1 + chol[1][1] * z2
                    # t-Copula: scale by sqrt(nu / chi2_nu)
                    chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(max(1, int(nu))))
                    scale = math.sqrt(max(chi2 / nu, 0.1))
                    x, y = x / scale, y / scale
                    scenarios.append([float(_std_norm_cdf(x)), float(_std_norm_cdf(y))])
            else:
                for _ in range(n_scenarios):
                    z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
                    x = z1
                    y = rho * z1 + math.sqrt(max(1 - rho * rho, 1e-15)) * z2
                    scenarios.append([float(_std_norm_cdf(x)), float(_std_norm_cdf(y))])
            return scenarios

        # Archimedean copulas: exact conditional sampling
        for _ in range(n_scenarios):
            u, w = rng.random(), rng.random()
            if ct == CopulaType.CLAYTON:
                theta = float(fit.params["theta"])
                v = (u ** (-theta / (theta + 1)) * (w ** (-theta / (theta + 1)) - 1) + 1) ** (-1.0 / theta)
            elif ct == CopulaType.GUMBEL:
                theta = float(fit.params["theta"])
                v = CopulaEngine._h_inverse(fit, u, w)
            elif ct == CopulaType.ROTATED_GUMBEL:
                theta = float(fit.params["theta"])
                v = CopulaEngine._h_inverse(fit, u, w)
            elif ct == CopulaType.FRANK:
                theta = float(fit.params["theta"])
                if abs(theta) > 1e-10:
                    et = math.exp(theta)
                    v = -1.0 / theta * math.log(1 + w * (et - 1) / (w + (1 - w) * math.exp(theta * u)))
                else:
                    v = w
            else:
                v = w
            scenarios.append([u, max(0.0, min(v, 1.0))])
        return scenarios

    # ============================================================
    # Tail dependence matrix
    # ============================================================

    @staticmethod
    def tail_dependence_matrix(rmat, method="nonparametric"):
        """Pairwise tail dependence coefficients."""
        n = len(rmat)
        result = [[Decimal("1") if i == j else Decimal("0") for j in range(n)] for i in range(n)]
        if method == "copula_fit":
            return CopulaEngine._tail_dep_via_copula(rmat, n)
        for i in range(n):
            for j in range(i + 1, n):
                ld = CopulaEngine._nonpar_lower_tail(rmat[i], rmat[j], Decimal("0.05"))
                result[i][j] = result[j][i] = ld
        return result

    @staticmethod
    def _tail_dep_via_copula(rmat, n):
        result = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                pair_rmat = [rmat[i], rmat[j]]
                n_obs = min(len(r) for r in pair_rmat)
                po = CopulaEngine._to_pseudo_obs(pair_rmat, 2, n_obs)
                fit_c = CopulaEngine._fit_clayton(po, n_obs)
                fit_g = CopulaEngine._fit_gumbel(po, n_obs)
                lt = fit_c.lower_tail_dep if fit_c else Decimal("0")
                ut = fit_g.upper_tail_dep if fit_g else Decimal("0")
                result[i][j] = result[j][i] = max(lt, ut)
        return result

    @staticmethod
    def _nonpar_lower_tail(x, y, q):
        xq = sorted(x)[int(len(x) * float(q))]
        yq = sorted(y)[int(len(y) * float(q))]
        both = sum(1 for xi, yi in zip(x, y, strict=False) if xi <= xq and yi <= yq)
        xc = sum(1 for xi in x if xi <= xq)
        return safe_divide(both, xc)

    # ============================================================
    # Portfolio tail loss
    # ============================================================

    @staticmethod
    def portfolio_tail_loss(fit, weights, n_scenarios=10000, confidence=Decimal("0.99")):
        scenarios = CopulaEngine.simulate(fit, n_scenarios)
        n_assets = len(weights)
        losses = [float(sum(weights[i] * Decimal(str(-s[i]))
                           for i in range(min(n_assets, len(s))))) for s in scenarios]
        losses.sort(reverse=True)
        idx = int(len(losses) * float(Decimal("1") - confidence))
        return Decimal(str(round(sum(losses[:idx]) / max(1, idx), 8)))

    # ============================================================
    # Compare all
    # ============================================================

    @staticmethod
    def compare_all(returns_matrix):
        """Fit all Copula types and return comparison summary."""
        all_types = [
            CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.ROTATED_GUMBEL,
            CopulaType.T, CopulaType.FRANK, CopulaType.GAUSSIAN,
        ]
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        pseudo_obs = CopulaEngine._to_pseudo_obs(returns_matrix, n_assets, n_obs)
        results = {}
        for ct in all_types:
            fit = CopulaEngine._fit_single(ct, pseudo_obs, returns_matrix, n_assets, n_obs)
            if fit:
                fit = CopulaEngine._gof_cramer_von_mises(fit, pseudo_obs, n_obs, 100)
                results[ct.value] = {
                    "params": {k: str(v) for k, v in fit.params.items()},
                    "lower_tail_dep": str(fit.lower_tail_dep),
                    "upper_tail_dep": str(fit.upper_tail_dep),
                    "log_likelihood": round(fit.log_likelihood, 2),
                    "aic": round(fit.aic, 2),
                    "cvm_statistic": round(fit.cvm_statistic, 4),
                    "cvm_pvalue": round(fit.cvm_pvalue, 4),
                    "gof_pass": fit.cvm_pvalue > 0.05,
                }
        return results


# -- Module-level bivariate normal CDF (reusable) --

def _bivariate_normal_cdf(x, y, rho):
    """Bivariate standard normal CDF via 12-point Gauss-Legendre quadrature."""
    if abs(rho) > 1.0:
        rho = min(max(rho, -1.0), 1.0)
    if math.isinf(x):
        return _std_norm_cdf(y) if x > 0 else 0.0
    if math.isinf(y):
        return _std_norm_cdf(x) if y > 0 else 0.0
    if abs(rho) < 1e-10:
        return _std_norm_cdf(x) * _std_norm_cdf(y)
    if abs(rho) > 0.9999:
        return min(_std_norm_cdf(x), _std_norm_cdf(y))

    total = 0.0
    for gi, wi in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
        t = rho * (1.0 + gi) / 2.0
        denom = 1.0 - t * t
        if denom <= 1e-15:
            continue
        exponent = -(x * x + y * y - 2.0 * t * x * y) / (2.0 * denom)
        if exponent < -700:
            continue
        total += wi * math.exp(exponent) / math.sqrt(denom)

    return max(0.0, min(1.0, _std_norm_cdf(x) * _std_norm_cdf(y) + (rho / 2.0) * total / (2.0 * math.pi)))
