"""
EVT 极值理论引擎 — GPD/POT/GEV 尾部风险建模 (v4.1).

纯计算层：无状态、无 IO、无数据库。
替换正态 VaR，对 A 股厚尾特征进行精确尾部估计。

v4.1 增强:
    - MLE GPD 拟合 (Newton-Raphson 迭代, 以 PWM 为初值)
    - Profile Likelihood 置信区间
    - 自动阈值选择 (Mean Excess Plot + 稳定性准则)
    - GEV 分布拟合 (Block Maxima)

数学基础:
    Pickands-Balkema-de Haan: 超额损失收敛于 GPD
    GPD CDF: G_{ξ,β}(y) = 1 - (1 + ξy/β)^{-1/ξ}  (ξ ≠ 0)
    VaR_α = u + (β/ξ) * ((n/N_u * (1-α))^{-ξ} - 1)
    ES_α  = VaR_α/(1-ξ) + (β - ξ*u)/(1-ξ)        (ξ < 1)
    GEV: G(x) = exp(-(1 + ξ(x-μ)/σ)^{-1/ξ})       (ξ ≠ 0)
    Return Level: z_p = μ - σ/ξ * (1 - (-ln(1-p))^{-ξ})

Usage:
    from jingsuan import EVTEngine
    fit = EVTEngine.fit_gpd(returns)
    var = EVTEngine.tail_var(fit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jingsuan.gev_engine import GEVFitResult  # noqa: F811

from shuju.utils import safe_divide, safe_mean

# ── Dataclasses ───────────────────────────────────────────

@dataclass
class EVTFitResult:
    """GPD 拟合结果 (v4.1 扩展)."""
    xi: Decimal          # 形状参数 (ξ > 0 = 厚尾)
    beta: Decimal        # 尺度参数
    threshold: Decimal   # POT 阈值
    n_total: int
    n_exceedances: int
    hill_xi: Decimal
    method: str = "pwm"  # "pwm" | "mle"
    xi_se: Decimal = Decimal("0")     # ξ 的标准误 (MLE)
    beta_se: Decimal = Decimal("0")   # β 的标准误 (MLE)
    convergence: bool = False         # MLE 是否收敛

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


@dataclass
class ProfileLikelihoodCI:
    """Profile Likelihood 置信区间."""
    parameter: str                      # "xi" | "beta" | "var_99" | "var_999"
    estimate: Decimal
    lower_95: Decimal
    upper_95: Decimal


# ── Constants ─────────────────────────────────────────────

_NORMAL_Z = {
    Decimal("0.95"): Decimal("1.645"),
    Decimal("0.99"): Decimal("2.326"),
    Decimal("0.999"): Decimal("3.090"),
}
_DEFAULT_LEVELS = [Decimal("0.95"), Decimal("0.99"), Decimal("0.999")]

# Chi-squared 0.95 quantile for 1 df (Profile Likelihood)
_CHI2_95_1DF = 3.8414588


# ── EVT Engine ────────────────────────────────────────────

class EVTEngine:
    """极值理论引擎 — 所有方法为静态方法，纯计算。"""

    # ═══════════════════════════════════════════════════════
    # GPD Fitting
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def fit_gpd(
        returns: list[Decimal],
        threshold_quantile: Decimal = Decimal("0.90"),
        method: str = "mle",
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> EVTFitResult:
        """POT 方法拟合 GPD.

        Args:
            returns: 日收益率序列
            threshold_quantile: POT 分位数 (默认 0.90)
            method: "pwm" (概率加权矩) | "mle" (最大似然, Newton-Raphson)
            max_iter: MLE 最大迭代次数
            tol: MLE 收敛容差
        """
        n = len(returns)
        if n < 50:
            raise ValueError(f"EVT requires >=50 observations, got {n}")

        losses = sorted([-r for r in returns if r < 0], reverse=True)
        if len(losses) < 10:
            losses = sorted([-r for r in returns], reverse=True)

        idx = max(10, min(int(len(losses) * float(threshold_quantile)), len(losses) - 10))
        threshold = losses[idx]
        exceedances = [x - threshold for x in losses[:idx] if x > threshold]

        # PWM 作为初值
        xi_pwm, beta_pwm = EVTEngine._pwm_gpd(exceedances)
        hill_xi = EVTEngine.hill_estimator(returns)

        if method == "pwm":
            return EVTFitResult(
                xi=xi_pwm, beta=beta_pwm, threshold=threshold,
                n_total=n, n_exceedances=len(exceedances),
                hill_xi=hill_xi, method="pwm",
            )

        # MLE via Newton-Raphson
        xi_mle, beta_mle, xi_se, beta_se, converged = EVTEngine._mle_gpd(
            exceedances, xi_pwm, beta_pwm, max_iter, tol
        )

        return EVTFitResult(
            xi=xi_mle, beta=beta_mle, threshold=threshold,
            n_total=n, n_exceedances=len(exceedances),
            hill_xi=hill_xi, method="mle",
            xi_se=xi_se, beta_se=beta_se, convergence=converged,
        )

    @staticmethod
    def _pwm_gpd(exceedances: list[Decimal]) -> tuple[Decimal, Decimal]:
        """PWM (Probability-Weighted Moments) GPD parameter estimation.

        Math:
            b_r = E[X * (1 - F(X))^r]
            b_0 = mean(X)
            b_1 = (1/n) * Σ x_{(j)} * (n-1-j)/(n-1)   (survival weights)
            For GPD: ξ = 2 - b_0/(b_0 - 2b_1), β = b_0·(1-ξ)
        """
        n = len(exceedances)
        xs = sorted(exceedances)
        b0 = safe_mean(xs)
        b1 = Decimal("0")
        for j, x in enumerate(xs):
            # Survival-based weight: (n-1-j)/(n-1) for sorted ascending
            b1 += x * Decimal(n - 1 - j)
        b1 = safe_divide(b1, Decimal(max(1, n * (n - 1))))

        denom = b0 - 2 * b1
        if denom == 0:
            return Decimal("0"), b0

        xi = max(Decimal("-0.49"), min(Decimal("2") - safe_divide(b0, denom), Decimal("2.0")))
        beta = b0 * (Decimal("1") - xi)
        if beta <= 0:
            beta = b0
        return xi, beta

    @staticmethod
    def _mle_gpd(
        exceedances: list[Decimal],
        xi0: Decimal, beta0: Decimal,
        max_iter: int, tol: float,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, bool]:
        """MLE GPD 拟合 — Newton-Raphson 迭代.

        Log-likelihood (ξ ≠ 0):
            ℓ(ξ, β) = -n·ln(β) - (1/ξ + 1)·Σ ln(1 + ξ·y_i/β)

        Gradient & Hessian computed analytically.

        Returns:
            (xi, beta, xi_se, beta_se, converged)
        """
        n = len(exceedances)
        y = [float(x) for x in exceedances]
        if n < 5:
            return xi0, beta0, Decimal("0"), Decimal("0"), False

        xi = float(xi0)
        beta = float(beta0)

        # Bounds
        xi = max(-0.49, min(xi, 0.99))
        beta = max(1e-10, beta)

        converged = False
        for _ in range(max_iter):
            # Check constraint: 1 + xi*y_i/beta > 0 for all i
            z = [1 + xi * yi / beta for yi in y]
            if min(z) <= 1e-15:
                xi = xi * 0.5  # step back
                continue

            # --- Gradient (score) ---
            # dℓ/dξ = Σ ln(z_i) / ξ² - (1/ξ + 1) * Σ (y_i/β) / z_i
            # dℓ/dβ = -n/β + (1/ξ + 1) * Σ (ξ·y_i/β²) / z_i

            sum_ln_z = sum(math.log(zi) for zi in z)
            sum_y_over_beta_z = sum((yi / beta) / zi for yi, zi in zip(y, z, strict=False))
            sum_xi_y_over_beta2_z = sum((xi * yi / (beta * beta)) / zi for yi, zi in zip(y, z, strict=False))

            d_xi = sum_ln_z / (xi * xi) - (1.0 / xi + 1.0) * sum_y_over_beta_z
            d_beta = -n / beta + (1.0 / xi + 1.0) * sum_xi_y_over_beta2_z

            # --- Hessian ---
            # ∂²ℓ/∂ξ² = -2·Σ ln(z_i)/ξ³ + 2·(1/ξ+1)·Σ y_i/β·(1/z_i)
            #            + (1/ξ+1)·Σ (y_i/β)²/z_i² · (ξ/ξ)
            # Simplified numerical approach for robustness:

            h_xi_xi = 0.0
            h_xi_beta = 0.0
            h_beta_beta = 0.0
            for yi, zi in zip(y, z, strict=False):
                yb = yi / beta
                yb2 = yb * yb
                # ∂²ℓ/∂ξ² entries
                h_xi_xi += -(2.0 * math.log(zi) / (xi ** 3) if abs(xi) > 1e-10 else 0) \
                           + (2.0 / xi ** 2) * yb / zi \
                           + (1.0 / xi + 1.0) * yb2 / (zi * zi)
                # ∂²ℓ/∂ξ∂β
                h_xi_beta += (-1.0 / (xi * xi)) * yb / zi \
                             + (1.0 / xi + 1.0) * (yi / (beta * beta)) * (1.0 + xi * yb) / (zi * zi)
                # ∂²ℓ/∂β²
                h_beta_beta += 1.0 / (beta * beta) \
                               - (1.0 / xi + 1.0) * (2.0 * xi * yi / (beta ** 3)) / zi \
                               - (1.0 / xi + 1.0) * (xi * xi * yi * yi / (beta ** 4)) / (zi * zi)

            # Build Hessian and invert
            det = h_xi_xi * h_beta_beta - h_xi_beta * h_xi_beta
            if abs(det) < 1e-15:
                break

            inv_h_xi_xi = h_beta_beta / det
            inv_h_xi_beta = -h_xi_beta / det
            inv_h_beta_beta = h_xi_xi / det

            # Newton step
            delta_xi = inv_h_xi_xi * d_xi + inv_h_xi_beta * d_beta
            delta_beta = inv_h_xi_beta * d_xi + inv_h_beta_beta * d_beta

            # Line search (halve step if needed)
            step = 1.0
            for _ in range(10):
                xi_new = xi - step * delta_xi
                beta_new = beta - step * delta_beta
                if all(1 + xi_new * yi / beta_new > 1e-15 for yi in y) and beta_new > 0:
                    break
                step *= 0.5
            else:
                break

            xi_new = max(-0.49, min(xi_new, 0.99))
            beta_new = max(1e-10, beta_new)

            if abs(xi_new - xi) < tol and abs(beta_new - beta) < tol:
                xi, beta = xi_new, beta_new
                converged = True
                break

            xi, beta = xi_new, beta_new

        # Standard errors from inverse observed Fisher information
        xi_se = Decimal("0")
        beta_se = Decimal("0")
        if converged:
            try:
                # Recompute Hessian at MLE
                z_final = [1 + xi * yi / beta for yi in y]
                h_xi_xi_f = 0.0
                h_beta_beta_f = 0.0
                h_xi_beta_f = 0.0
                for yi, zi in zip(y, z_final, strict=False):
                    yb = yi / beta
                    yb2 = yb * yb
                    h_xi_xi_f += -(2.0 * math.log(zi) / (xi ** 3) if abs(xi) > 1e-10 else 0) \
                                 + (2.0 / xi ** 2) * yb / zi \
                                 + (1.0 / xi + 1.0) * yb2 / (zi * zi)
                    h_xi_beta_f += (-1.0 / (xi * xi)) * yb / zi \
                                   + (1.0 / xi + 1.0) * (yi / (beta * beta)) * (1.0 + xi * yb) / (zi * zi)
                    h_beta_beta_f += 1.0 / (beta * beta) \
                                     - (1.0 / xi + 1.0) * (2.0 * xi * yi / (beta ** 3)) / zi \
                                     - (1.0 / xi + 1.0) * (xi * xi * yi * yi / (beta ** 4)) / (zi * zi)

                det_f = h_xi_xi_f * h_beta_beta_f - h_xi_beta_f * h_xi_beta_f
                if abs(det_f) > 1e-15:
                    xi_se = Decimal(str(round(math.sqrt(abs(h_beta_beta_f / det_f)), 8)))
                    beta_se = Decimal(str(round(math.sqrt(abs(h_xi_xi_f / det_f)), 8)))
            except (ValueError, OverflowError, ZeroDivisionError):
                pass

        return Decimal(str(xi)), Decimal(str(beta)), xi_se, beta_se, converged

    # ═══════════════════════════════════════════════════════
    # Profile Likelihood Confidence Intervals
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def profile_likelihood_ci(
        returns: list[Decimal],
        fit: EVTFitResult | None = None,
        threshold_quantile: Decimal = Decimal("0.90"),
        parameter: str = "xi",
        var_level: Decimal = Decimal("0.99"),
        n_grid: int = 50,
    ) -> ProfileLikelihoodCI:
        """Profile Likelihood 置信区间.

        Args:
            returns: 日收益率序列
            fit: 已有的 EVTFitResult (None 时自动拟合 MLE)
            threshold_quantile: POT 阈值分位数
            parameter: "xi" | "beta" | "var_99" | "var_999"
            var_level: 当 parameter 为 "var_*" 时的置信水平
            n_grid: 网格搜索点数

        Returns:
            ProfileLikelihoodCI with 95% CI bounds.

        Theory:
            Under H₀: θ = θ₀, 2 * (ℓ(θ̂) - ℓ(θ₀)) ~ χ²(1)
            CI: {θ₀ : 2 * (ℓ_max - ℓ(θ₀)) ≤ χ²_{0.95}(1)}
        """
        if fit is None or fit.method != "mle":
            fit = EVTEngine.fit_gpd(returns, threshold_quantile, method="mle")

        losses = sorted([-r for r in returns if r < 0], reverse=True)
        if len(losses) < 10:
            losses = sorted([-r for r in returns], reverse=True)
        idx = max(10, min(int(len(losses) * float(threshold_quantile)), len(losses) - 10))
        threshold = fit.threshold
        exceedances = [x - threshold for x in losses[:idx] if x > threshold]
        y = [float(x) for x in exceedances]

        xi_hat = float(fit.xi)
        beta_hat = float(fit.beta)
        ll_max = EVTEngine._gpd_loglik(y, xi_hat, beta_hat)
        cutoff = ll_max - _CHI2_95_1DF / 2.0

        if parameter == "xi":
            estimate = fit.xi
            xi_grid = [xi_hat * (1.0 + (i - n_grid / 2) * 0.02) for i in range(n_grid + 1)]
            xi_grid = [max(-0.45, min(x, 0.95)) for x in xi_grid]
            ci_lower = xi_hat
            ci_upper = xi_hat
            for xi_g in xi_grid:
                beta_pf = EVTEngine._profile_beta_given_xi(y, xi_g, beta_hat)
                ll = EVTEngine._gpd_loglik(y, xi_g, beta_pf)
                if ll >= cutoff:
                    ci_lower = min(ci_lower, xi_g)
                    ci_upper = max(ci_upper, xi_g)

        elif parameter == "beta":
            estimate = fit.beta
            beta_grid = [beta_hat * (1.0 + (i - n_grid / 2) * 0.03) for i in range(n_grid + 1)]
            beta_grid = [max(1e-8, b) for b in beta_grid]
            ci_lower = beta_hat
            ci_upper = beta_hat
            for beta_g in beta_grid:
                xi_pf = EVTEngine._profile_xi_given_beta(y, beta_g, xi_hat)
                ll = EVTEngine._gpd_loglik(y, xi_pf, beta_g)
                if ll >= cutoff:
                    ci_lower = min(ci_lower, beta_g)
                    ci_upper = max(ci_upper, beta_g)

        elif parameter.startswith("var"):
            estimate = EVTEngine._var_from_params(
                Decimal(str(xi_hat)), Decimal(str(beta_hat)),
                fit.threshold, float(fit.n_total), float(fit.n_exceedances),
                var_level
            )
            xi_grid = [xi_hat * (1.0 + (i - n_grid / 2) * 0.02) for i in range(n_grid + 1)]
            xi_grid = [max(-0.45, min(x, 0.95)) for x in xi_grid]
            ci_lower = float(estimate)
            ci_upper = float(estimate)
            for xi_g in xi_grid:
                beta_pf = EVTEngine._profile_beta_given_xi(y, xi_g, beta_hat)
                ll = EVTEngine._gpd_loglik(y, xi_g, beta_pf)
                if ll >= cutoff:
                    var_v = EVTEngine._var_from_params(
                        Decimal(str(xi_g)), Decimal(str(beta_pf)),
                        fit.threshold, float(fit.n_total), float(fit.n_exceedances),
                        var_level
                    )
                    ci_lower = min(ci_lower, float(var_v))
                    ci_upper = max(ci_upper, float(var_v))
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

        return ProfileLikelihoodCI(
            parameter=parameter,
            estimate=estimate,
            lower_95=Decimal(str(round(ci_lower, 8))),
            upper_95=Decimal(str(round(ci_upper, 8))),
        )

    @staticmethod
    def _gpd_loglik(y: list[float], xi: float, beta: float) -> float:
        """GPD log-likelihood (ξ ≠ 0)."""
        len(y)
        ll = 0.0
        for yi in y:
            z = 1 + xi * yi / beta
            if z <= 0:
                return -1e20
            ll -= math.log(beta) + (1.0 / xi + 1.0) * math.log(z)
        return ll

    @staticmethod
    def _profile_beta_given_xi(y: list[float], xi: float, beta_init: float) -> float:
        """For fixed ξ, profile MLE of β: solve ∂ℓ/∂β = 0."""
        n = len(y)
        if abs(xi) < 1e-10:
            return sum(y) / n
        beta = beta_init
        for _ in range(50):
            z = [1 + xi * yi / beta for yi in y]
            if min(z) <= 1e-15:
                beta *= 0.5
                continue
            # Score for β: -n/β + (1/ξ+1) * Σ (ξ·y_i/β²) / z_i = 0
            # => β = (1/ξ+1)/n * Σ (ξ·y_i) / z_i
            sum_term = sum((xi * yi) / zi for yi, zi in zip(y, z, strict=False))
            beta_new = (1.0 / xi + 1.0) * sum_term / n
            if abs(beta_new - beta) < 1e-8:
                return max(1e-10, beta_new)
            beta = max(1e-10, beta_new)
        return max(1e-10, beta)

    @staticmethod
    def _profile_xi_given_beta(y: list[float], beta: float, xi_init: float) -> float:
        """For fixed β, profile MLE of ξ: solve ∂ℓ/∂ξ = 0."""
        len(y)
        xi = xi_init
        for _ in range(50):
            z = [1 + xi * yi / beta for yi in y]
            if min(z) <= 1e-15:
                xi *= 0.5
                continue
            # Score for ξ: Σ ln(z_i)/ξ² - (1/ξ+1)·Σ y_i/β / z_i = 0
            sum_ln_z = sum(math.log(zi) for zi in z)
            sum_yb = sum((yi / beta) / zi for yi, zi in zip(y, z, strict=False))
            # Solve numerically: f(ξ) = Σ ln(z)/ξ² - (1/ξ+1)·Σ yb = 0
            # Use Newton on ξ
            f_val = sum_ln_z / (xi * xi) - (1.0 / xi + 1.0) * sum_yb
            f_prime = -2.0 * sum_ln_z / (xi ** 3) \
                      + (1.0 / (xi * xi)) * sum_yb
            if abs(f_prime) < 1e-15:
                break
            xi_new = xi - f_val / f_prime
            xi_new = max(-0.45, min(xi_new, 0.95))
            if abs(xi_new - xi) < 1e-8:
                return xi_new
            xi = xi_new
        return xi

    @staticmethod
    def _var_from_params(
        xi: Decimal, beta: Decimal, u: Decimal,
        n_total: float, n_exc: float, level: Decimal,
    ) -> Decimal:
        """Compute VaR from GPD parameters."""
        tail_ratio = Decimal(str(n_exc / n_total)) if n_total > 0 else Decimal("0")
        p = tail_ratio * (Decimal("1") - level)
        if p <= 0:
            return u
        if xi == 0:
            return u + beta * (Decimal("1") / p).ln()
        return u + (beta / xi) * (p ** (-xi) - Decimal("1"))

    # ═══════════════════════════════════════════════════════
    # Auto Threshold Selection
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def auto_threshold(
        returns: list[Decimal],
        quantile_range: tuple[float, float] = (0.85, 0.98),
        n_points: int = 20,
    ) -> tuple[Decimal, dict]:
        """自动阈值选择 — Mean Excess Plot + 稳定性准则.

        Algo:
        1. 在 [q_low, q_high] 范围内扫描候选阈值
        2. 对每个阈值拟合 GPD，记录 ξ 和 β
        3. 选择 ξ 估计最稳定的最低阈值 (minimize variance of ξ
           over a window of adjacent thresholds)

        Args:
            returns: 日收益率序列
            quantile_range: 扫描的分位数范围
            n_points: 扫描点数

        Returns:
            (optimal_threshold, diagnostics_dict)
        """
        losses = sorted([-r for r in returns if r < 0], reverse=True)
        if len(losses) < 30:
            losses = sorted([-r for r in returns], reverse=True)

        n_total = len(losses)
        q_low, q_high = quantile_range
        thresholds = []
        xi_estimates = []
        beta_estimates = []
        n_exc_list = []

        for i in range(n_points):
            q = q_low + (q_high - q_low) * i / (n_points - 1)
            idx = max(5, min(int(n_total * q), n_total - 5))
            u = losses[idx]
            u_float = float(u)
            exc = [float(x) - u_float for x in losses[:idx] if float(x) > u_float]
            n_exc = len(exc)
            if n_exc < 5:
                continue

            xi_pwm, beta_pwm = EVTEngine._pwm_gpd(
                [Decimal(str(e)) for e in exc]
            )
            thresholds.append(u)
            xi_estimates.append(float(xi_pwm))
            beta_estimates.append(float(beta_pwm))
            n_exc_list.append(n_exc)

        if len(thresholds) < 3:
            idx = max(10, min(int(n_total * 0.90), n_total - 10))
            return losses[idx], {"error": "insufficient_thresholds"}

        # Stability: find range where ξ estimates are most stable
        # Use rolling window of 5 consecutive thresholds
        window = min(5, len(thresholds) // 2)
        best_stability = float("inf")
        best_idx = window // 2

        for i in range(len(thresholds) - window + 1):
            xi_window = xi_estimates[i:i + window]
            xi_mean = sum(xi_window) / len(xi_window)
            xi_variance = sum((x - xi_mean) ** 2 for x in xi_window) / len(xi_window)
            # Penalize too few exceedances
            n_exc_penalty = max(0, 30 - n_exc_list[i]) * 0.0001
            stability = xi_variance + n_exc_penalty
            if stability < best_stability:
                best_stability = stability
                best_idx = i

        optimal_threshold = Decimal(str(thresholds[best_idx]))

        diagnostics = {
            "thresholds_scanned": [Decimal(str(round(t, 6))) for t in thresholds],
            "xi_estimates": [Decimal(str(round(x, 6))) for x in xi_estimates],
            "n_exceedances_per_threshold": n_exc_list,
            "selected_index": best_idx,
            "stability_score": Decimal(str(round(best_stability, 8))),
            "mean_excess": EVTEngine._mean_excess_plot(losses, n_total),
        }

        return optimal_threshold, diagnostics

    @staticmethod
    def _mean_excess_plot(losses: list[Decimal], n_total: int) -> dict:
        """Mean Excess Function: e(u) = E[X-u | X>u].

        For GPD with ξ < 1: e(u) = (β + ξ·u) / (1-ξ), linear in u.
        Positive slope → ξ > 0 (heavy-tailed).
        """
        n_points = min(30, len(losses) // 5)
        result = {}
        for i in range(1, n_points):
            idx = i * len(losses) // n_points
            u = losses[idx]
            exc = [float(x) - float(u) for x in losses[:idx] if float(x) > float(u)]
            if exc:
                mean_exc = sum(exc) / len(exc)
                result[str(float(round(float(u), 6)))] = Decimal(str(round(mean_exc, 8)))
        return result

    # ═══════════════════════════════════════════════════════
    # GEV Distribution (Block Maxima)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def fit_gev(
        returns: list[Decimal],
        block_size: int = 21,   # ~monthly blocks for daily data
        method: str = "pwm",
    ) -> GEVFitResult:
        """GEV 分布拟合 (Block Maxima 方法).

        Args:
            returns: 日收益率 (正=盈利, 负=亏损)
            block_size: 块大小 (日, 默认 21 = 月)
            method: "pwm" | "mle"

        Returns:
            GEVFitResult with (mu, sigma, xi, return_levels)
        """
        from jingsuan.gev_engine import GEVEngine
        # Negate: model maxima of losses
        losses = [-float(r) for r in returns]
        # Block maxima
        n = len(losses)
        n_blocks = n // block_size
        if n_blocks < 10:
            raise ValueError(f"Need >= 10 blocks, got {n_blocks} (n={n}, block={block_size})")
        block_maxima = []
        for i in range(n_blocks):
            block = losses[i * block_size:(i + 1) * block_size]
            block_maxima.append(Decimal(str(max(block))))
        return GEVEngine.fit(block_maxima, method=method)

    # ═══════════════════════════════════════════════════════
    # Hill Estimator
    # ═══════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════
    # VaR / ES
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def tail_var(
        fit: EVTFitResult | GEVFitResult,  # type: ignore[name-defined]  # noqa: F821
        levels: list[Decimal] | None = None,
    ) -> EVTVaRResult:
        """GPD VaR/ES 计算."""
        if levels is None:
            levels = _DEFAULT_LEVELS

        # Handle GEV fit result
        from jingsuan.gev_engine import GEVFitResult

        if isinstance(fit, GEVFitResult):
            mu, sigma, xi = fit.mu, fit.sigma, fit.xi
            def _var_gev(alpha: Decimal) -> Decimal:
                # GEV quantile: z_p = μ - σ/ξ·(1 - (-ln(p))^{-ξ})  (ξ ≠ 0)
                #              z_p = μ - σ·ln(-ln(p))              (ξ = 0)
                neg_ln_p = (-alpha.ln())  # -ln(p), p = alpha = confidence level
                if xi == 0:
                    return mu - sigma * neg_ln_p.ln()
                return mu - (sigma / xi) * (Decimal("1") - neg_ln_p ** (-xi))

            def _es_gev(alpha: Decimal) -> Decimal:
                # ES for GEV via numerical integration of VaR over tail
                n_grid = 100
                p_start = float(alpha)
                step = (0.9999 - p_start) / n_grid
                total = Decimal("0")
                for j in range(n_grid):
                    p_j = p_start + j * step
                    total += _var_gev(Decimal(str(p_j)))
                return total / Decimal(str(n_grid))

            results = {a: (_var_gev(a), _es_gev(a)) for a in levels}
            return EVTVaRResult(
                var_95=results[Decimal("0.95")][0],
                var_99=results[Decimal("0.99")][0],
                var_999=results[Decimal("0.999")][0],
                es_95=results[Decimal("0.95")][1],
                es_99=results[Decimal("0.99")][1],
                es_999=results[Decimal("0.999")][1],
            )

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

    # ═══════════════════════════════════════════════════════
    # Model Comparison
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def compare_models(returns: list[Decimal]) -> dict:
        """正态 vs EVT(GPD) vs EVT(GEV) vs 历史法 VaR 对比."""
        n = len(returns)
        fit_gpd = EVTEngine.fit_gpd(returns)
        evt_gpd = EVTEngine.tail_var(fit_gpd)

        # GEV
        try:
            fit_gev = EVTEngine.fit_gev(returns)
            evt_gev = EVTEngine.tail_var(fit_gev)
            has_gev = True
        except Exception:
            has_gev = False

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

            entry = {
                "normal": (n_var, n_es),
                "evt_gpd": (getattr(evt_gpd, var_attr), getattr(evt_gpd, es_attr)),
                "historical": (h_var, h_es),
            }
            if has_gev:
                entry["evt_gev"] = (getattr(evt_gev, var_attr), getattr(evt_gev, es_attr))
            result[str(float(cl))] = entry
        return result


# ── Helpers ────────────────────────────────────────────────

def _normal_es_factor(z: Decimal) -> Decimal:
    """N(0,1) 在 z 处的 ES 缩放因子 φ(z)/Φ(-z)."""
    from math import exp as _exp
    phi = Decimal(str(_exp(-float(z * z) / 2) / 2.5066282746310002))
    phi_neg = Decimal("0.5") * (Decimal("1") - _erf_approx(float(z) / 1.4142135623730951))
    return safe_divide(phi, phi_neg)


def _erf_approx(x: float) -> Decimal:
    """误差函数的数值近似."""
    import math
    return Decimal(str(math.erf(x)))
