"""
EVT 极值理论引擎 — GPD/POT 尾部风险建模 (v4.0).

纯计算层：无状态、无 IO、无数据库。
替换正态 VaR，对 A 股厚尾特征进行精确尾部估计。

数学基础:
    Pickands-Balkema-de Haan: 超额损失收敛于 GPD
    VaR_α = u + (β/ξ) * ((n/N_u * (1-α))^(-ξ) - 1)   ξ ≠ 0
    ES_α  = VaR_α/(1-ξ) + (β - ξ*u)/(1-ξ)            ξ < 1
    Hill_ξ = (1/k) * Σ ln(X_{(i)}/X_{(k+1)})

Usage:
    from jingsuan.evt_engine import EVTEngine
    fit = EVTEngine.fit_gpd(returns)
    var = EVTEngine.tail_var(fit)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide, safe_mean


@dataclass
class EVTFitResult:
    """GPD 拟合结果."""
    xi: Decimal          # 形状参数 (ξ > 0 = 厚尾)
    beta: Decimal        # 尺度参数
    threshold: Decimal   # POT 阈值
    n_total: int
    n_exceedances: int
    hill_xi: Decimal

    @property
    def is_heavy_tailed(self) -> bool:
        """True if ξ > 0 (thicker tails than normal)."""
        return self.xi > Decimal("0")


@dataclass
class EVTVaRResult:
    """EVT VaR / Expected Shortfall."""
    var_95: Decimal
    var_99: Decimal
    var_999: Decimal
    es_95: Decimal
    es_99: Decimal
    es_999: Decimal


_NORMAL_Z = {
    Decimal("0.95"): Decimal("1.645"),
    Decimal("0.99"): Decimal("2.326"),
    Decimal("0.999"): Decimal("3.090"),
}
_DEFAULT_LEVELS = [Decimal("0.95"), Decimal("0.99"), Decimal("0.999")]


class EVTEngine:
    """极值理论引擎 — 所有方法为静态方法，纯计算。"""

    @staticmethod
    def fit_gpd(
        returns: list[Decimal],
        threshold_quantile: Decimal = Decimal("0.90"),
    ) -> EVTFitResult:
        """POT 方法拟合 GPD."""
        n = len(returns)
        if n < 50:
            raise ValueError(f"EVT requires >=50 observations, got {n}")

        losses = sorted([-r for r in returns if r < 0], reverse=True)
        if len(losses) < 10:
            losses = sorted([-r for r in returns], reverse=True)

        idx = max(10, min(int(len(losses) * float(threshold_quantile)), len(losses) - 10))
        threshold = losses[idx]
        exceedances = [x - threshold for x in losses[:idx] if x > threshold]

        xi, beta = EVTEngine._pwm_gpd(exceedances)
        hill_xi = EVTEngine.hill_estimator(returns)

        return EVTFitResult(
            xi=xi, beta=beta, threshold=threshold,
            n_total=n, n_exceedances=len(exceedances),
            hill_xi=hill_xi,
        )

    @staticmethod
    def _pwm_gpd(exceedances: list[Decimal]) -> tuple[Decimal, Decimal]:
        """PWM (概率加权矩) 估计 GPD 参数."""
        n = len(exceedances)
        xs = sorted(exceedances)
        b0 = safe_mean(xs)
        b1 = Decimal("0")
        for j, x in enumerate(xs):
            b1 += x * Decimal(j)
        b1 = safe_divide(b1, Decimal(max(1, n * (n - 1))))

        denom = b0 - 2 * b1
        if denom == 0:
            return Decimal("0"), b0

        xi = max(Decimal("-0.49"), min(Decimal("2") - safe_divide(b0, denom), Decimal("0.99")))
        beta = max(safe_divide(2 * b0 * b1, denom), b0)
        return xi, beta

    @staticmethod
    def hill_estimator(
        returns: list[Decimal],
        tail_fraction: Decimal = Decimal("0.10"),
    ) -> Decimal:
        """Hill 非参数尾指数估计 ξ = (1/k) Σ ln(X(i)/X(k+1))."""
        losses = sorted([-r for r in returns if r < 0], reverse=True)
        if len(losses) < 20:
            losses = sorted([-r for r in returns], reverse=True)

        k = max(5, min(int(len(losses) * float(tail_fraction)), len(losses) - 1))
        if losses[k] == 0:
            return Decimal("0")

        hill = Decimal("0")
        for i in range(k):
            ratio = safe_divide(losses[i], losses[k])
            if ratio > 0:
                hill += ratio.ln()
        return safe_divide(hill, Decimal(k))

    @staticmethod
    def tail_var(
        fit: EVTFitResult,
        levels: list[Decimal] | None = None,
    ) -> EVTVaRResult:
        """GPD VaR/ES 计算."""
        if levels is None:
            levels = _DEFAULT_LEVELS

        xi, beta, u = fit.xi, fit.beta, fit.threshold
        tail_ratio = Decimal(fit.n_exceedances) / Decimal(fit.n_total)

        def _var(alpha: Decimal) -> Decimal:
            p = tail_ratio * (1 - alpha)
            if p <= 0:
                return u
            if xi == 0:
                return u + beta * (1 / p).ln()
            return u + (beta / xi) * (p ** (-xi) - 1)

        def _es(alpha: Decimal) -> Decimal:
            v = _var(alpha)
            if xi >= 1:
                return v
            return v / (1 - xi) + (beta - xi * u) / (1 - xi) if xi != 0 else v + beta

        results = {a: (_var(a), _es(a)) for a in levels}
        return EVTVaRResult(
            var_95=results[Decimal("0.95")][0],
            var_99=results[Decimal("0.99")][0],
            var_999=results[Decimal("0.999")][0],
            es_95=results[Decimal("0.95")][1],
            es_99=results[Decimal("0.99")][1],
            es_999=results[Decimal("0.999")][1],
        )

    @staticmethod
    def compare_models(returns: list[Decimal]) -> dict:
        """正态 vs EVT vs 历史法 VaR 对比。"""
        n = len(returns)
        fit = EVTEngine.fit_gpd(returns)
        evt = EVTEngine.tail_var(fit)

        mu = safe_mean(returns)
        sigma = (sum((r - mu) ** 2 for r in returns) / Decimal(max(1, n - 1))).sqrt()

        losses = sorted([-r for r in returns])
        result = {}
        for cl, var_attr, es_attr in [
            (Decimal("0.95"), "var_95", "es_95"),
            (Decimal("0.99"), "var_99", "es_99"),
            (Decimal("0.999"), "var_999", "es_999"),
        ]:
            z = _NORMAL_Z[cl]
            n_var = mu - z * sigma
            n_es = mu - sigma * _normal_es_factor(z)

            idx = min(int(n * float(cl)), n - 1)
            h_var = losses[idx]
            h_es = safe_mean([x for x in losses if x >= h_var]) or h_var

            result[str(float(cl))] = {
                "normal": (n_var, n_es),
                "evt": (getattr(evt, var_attr), getattr(evt, es_attr)),
                "historical": (h_var, h_es),
            }
        return result


def _normal_es_factor(z: Decimal) -> Decimal:
    """N(0,1) 在 z 处的 ES 缩放因子 φ(z)/Φ(-z)."""
    from math import exp as _exp
    phi = Decimal(str(_exp(-float(z * z) / 2) / 2.5066282746310002))
    # Φ(-z) 近似
    phi_neg = Decimal("0.5") * (Decimal("1") - _erf_approx(float(z) / 1.4142135623730951))
    return safe_divide(phi, phi_neg)


def _erf_approx(x: float) -> Decimal:
    """误差函数的数值近似."""
    import math
    return Decimal(str(math.erf(x)))
