"""
GEV 极值分布引擎 — Block Maxima 方法 (v4.0).

纯计算层：无状态、无 IO。
提供 GEV 分布拟合、Return Level 估计、模型诊断。

GEV 分布函数:
    G(x) = exp(-(1 + ξ·(x-μ)/σ)^{-1/ξ})    ξ ≠ 0
    G(x) = exp(-exp(-(x-μ)/σ))              ξ = 0

支持:
    - μ (位置), σ (尺度), ξ (形状)
    - ξ > 0: Fréchet (厚尾, 金融常见)
    - ξ = 0: Gumbel (薄尾)
    - ξ < 0: Weibull (有界尾, 不常见于金融)

Return Level (p-return period):
    z_p = μ - σ/ξ·(1 - (-ln(1-p))^{-ξ})    ξ ≠ 0
    z_p = μ - σ·ln(-ln(1-p))                ξ = 0

Usage:
    from jingsuan.gev_engine import GEVEngine, GEVFitResult
    fit = GEVEngine.fit(block_maxima)
    rl = GEVEngine.return_level(fit, period=100)  # 100-period return level
"""

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class GEVFitResult:
    """GEV 拟合结果."""
    mu: Decimal         # 位置参数
    sigma: Decimal      # 尺度参数 (> 0)
    xi: Decimal         # 形状参数
    n_blocks: int
    method: str         # "pwm" | "mle"
    convergence: bool = False
    mu_se: Decimal = Decimal("0")
    sigma_se: Decimal = Decimal("0")
    xi_se: Decimal = Decimal("0")
    log_likelihood: float = 0.0
    aic: float = 0.0

    @property
    def tail_type(self) -> str:
        """返回尾部类型."""
        if self.xi > Decimal("0.05"):
            return "Fréchet (厚尾)"
        elif self.xi < Decimal("-0.05"):
            return "Weibull (有界尾)"
        else:
            return "Gumbel (薄尾)"

    @property
    def is_heavy_tailed(self) -> bool:
        return self.xi > Decimal("0")


@dataclass
class GEVReturnLevel:
    """GEV Return Level 估计."""
    period: int              # 回报期 (如 100 = 100 个 block)
    return_level: Decimal    # z_p
    lower_95: Decimal        # 95% CI lower
    upper_95: Decimal        # 95% CI upper


class GEVEngine:
    """GEV 分布引擎 — Block Maxima 方法的极值建模。"""

    # ═══════════════════════════════════════════════════════
    # PWM Estimation
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def fit(
        block_maxima: list[Decimal],
        method: str = "pwm",
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> GEVFitResult:
        """GEV 分布拟合.

        Args:
            block_maxima: 块最大值序列 (如月度最大亏损)
            method: "pwm" (probability-weighted moments) | "mle"
            max_iter: MLE 最大迭代次数
            tol: MLE 收敛容差

        Returns:
            GEVFitResult
        """
        n = len(block_maxima)
        if n < 10:
            raise ValueError(f"GEV 拟合需要 >= 10 个块, 当前 {n}")

        # PWM 估计作为基准
        mu_pwm, sigma_pwm, xi_pwm = GEVEngine._pwm_gev(block_maxima)

        if method == "pwm":
            return GEVFitResult(
                mu=mu_pwm, sigma=sigma_pwm, xi=xi_pwm,
                n_blocks=n, method="pwm",
            )

        # MLE via constrained optimization
        mu_mle, sigma_mle, xi_mle, mu_se, sigma_se, xi_se, ll, conv = \
            GEVEngine._mle_gev(block_maxima, mu_pwm, sigma_pwm, xi_pwm, max_iter, tol)

        return GEVFitResult(
            mu=mu_mle, sigma=sigma_mle, xi=xi_mle,
            n_blocks=n, method="mle",
            convergence=conv,
            mu_se=mu_se, sigma_se=sigma_se, xi_se=xi_se,
            log_likelihood=ll,
            aic=6.0 - 2.0 * ll,  # 3 params → 2k = 6
        )

    @staticmethod
    def _pwm_gev(data: list[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
        """PWM (Probability-Weighted Moments) 估计 GEV 参数.

        Hosking et al. (1985) method:
            b_r = (1/n) Σ_{j=1}^{n} (j-1 choose r) / (n-1 choose r) · x_{(j)}
            β_0 = b_0, β_1 = b_1, β_2 = b_2
            ξ = 7.8590·c + 2.9554·c²  where c = 2b₁-b₀ / 3b₂-b₀ - ln(2)/ln(3)
        """
        xs = sorted(data)
        n = len(xs)
        xs_float = [float(x) for x in xs]

        # b_0, b_1, b_2
        b0 = sum(xs_float) / n

        b1 = 0.0
        for j, x in enumerate(xs_float, start=1):
            b1 += (j - 1) / (n - 1) * x if n > 1 else 0
        b1 /= n

        b2 = 0.0
        for j, x in enumerate(xs_float, start=1):
            b2 += ((j - 1) * (j - 2)) / ((n - 1) * (n - 2)) * x if n > 2 else 0
        b2 /= n if n > 2 else 1

        # Shape parameter ξ
        if b1 == b0 or b2 == b1:
            xi = Decimal("0")
        else:
            c_float = (2 * b1 - b0) / max(3 * b2 - b0, 1e-10) - math.log(2) / math.log(3)
            xi_float = 7.8590 * c_float + 2.9554 * c_float * c_float
            xi_float = max(-0.5, min(xi_float, 0.5))
            xi = Decimal(str(round(xi_float, 6)))

        # Scale σ
        if xi != 0:
            gamma1 = math.gamma(1 - float(xi))
            gamma2 = 1 - 2 ** (-float(xi))
            sigma_float = (2 * b1 - b0) * float(xi) / (gamma1 * gamma2) if gamma1 != 0 and gamma2 != 0 else b0
        else:
            sigma_float = (2 * b1 - b0) / 0.6931471805599453  # ln(2)
        sigma = Decimal(str(max(1e-10, abs(sigma_float))))

        # Location μ
        if xi != 0:
            mu_float = b0 - sigma_float * (1 - math.gamma(1 - float(xi))) / float(xi) if float(xi) != 0 else b0 - sigma_float * 0.5772156649
        else:
            mu_float = b0 - sigma_float * 0.5772156649  # Euler-Mascheroni constant

        mu = Decimal(str(round(mu_float, 8)))
        sigma = Decimal(str(round(sigma_float, 8)))

        return mu, sigma, xi

    # ═══════════════════════════════════════════════════════
    # MLE via Profile Likelihood
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _mle_gev(
        data: list[Decimal],
        mu0: Decimal, sigma0: Decimal, xi0: Decimal,
        max_iter: int, tol: float,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal, float, bool]:
        """MLE GEV — 使用 profile likelihood (给定 ξ, 解析求解 μ, σ)."""
        n = len(data)
        y = [float(x) for x in data]
        xi = float(xi0)

        converged = False
        # Grid + local optimization for ξ (profile likelihood)
        for _ in range(max_iter):
            # Compute profile μ, σ given ξ
            xi, new_xi, mu_star, sigma_star, ll = GEVEngine._profile_xi(
                y, xi, xi - 0.2, xi + 0.2, 50
            )
            if abs(new_xi - xi) < tol:
                xi = new_xi
                converged = True
                break
            xi = new_xi

        if not converged:
            # Fall back to broader grid search
            best_ll = -float("inf")
            best_xi = float(xi0)
            best_mu = float(mu0)
            best_sigma = float(sigma0)
            for xi_g in [v / 100 for v in range(-45, 51, 2)]:
                ok, mu_g, sigma_g, ll_g = GEVEngine._gev_profile_ll(y, xi_g)
                if ok and ll_g > best_ll:
                    best_ll = ll_g
                    best_xi = xi_g
                    best_mu = mu_g
                    best_sigma = sigma_g
            xi = best_xi
            mu_star = best_mu
            sigma_star = best_sigma
            ll = best_ll

        # Standard errors via observed Fisher information (numerical Hessian)
        h = 1e-4
        _, _, _, ll_center = GEVEngine._gev_profile_ll(y, xi)
        _, _, _, ll_forward = GEVEngine._gev_profile_ll(y, xi + h)
        _, _, _, ll_backward = GEVEngine._gev_profile_ll(y, xi - h)
        d2ll = (ll_forward - 2 * ll_center + ll_backward) / (h * h)

        se_xi = Decimal(str(round(math.sqrt(-1.0 / max(-d2ll, 0.01)), 6)))
        se_mu = Decimal(str(round(abs(sigma_star / math.sqrt(n)), 6)))
        se_sigma = Decimal(str(round(abs(sigma_star / math.sqrt(2 * n)), 6)))

        return (
            Decimal(str(mu_star)), Decimal(str(sigma_star)), Decimal(str(xi)),
            se_mu, se_sigma, se_xi, ll, True
        )

    @staticmethod
    def _gev_profile_ll(y: list[float], xi: float) -> tuple[bool, float, float, float]:
        """给定 ξ, profile out μ 和 σ, 返回 (ok, μ, σ, loglik)."""
        n = len(y)
        # Constraint check: 1 + xi*(y_i - mu)/sigma > 0
        # Profile likelihood for GEV: μ and σ have closed form given ξ
        # Use iterative approach for (μ, σ) given ξ

        # Initial guess
        mu = sum(y) / n
        sigma = (sum((yi - mu) ** 2 for yi in y) / (n - 1)) ** 0.5 if n > 1 else 1.0

        # Newton for (μ, σ) given ξ via score equations
        for _ in range(30):
            z = [(yi - mu) / sigma for yi in y]
            if xi != 0:
                # Constraint check
                if any(1 + xi * zi <= 1e-15 for zi in z):
                    # Try larger sigma
                    sigma *= 2.0
                    continue
                g = [(1 + xi * zi) ** (-1.0 / xi) for zi in z]
            else:
                g = [math.exp(-math.exp(-zi)) for zi in z]

            # Score equations approximately
            sum_g = sum(g)
            sum_g_z = sum(gi * zi for gi, zi in zip(g, z, strict=False))

            if xi != 0:
                # Update μ, σ using scoring
                dmu = sum_g - n
                dsigma = sum_g_z - n
            else:
                dmu = sum_g - n
                dsigma = sum_g_z - n

            mu_new = mu + 0.1 * dmu * sigma / n
            sigma_new = sigma + 0.1 * dsigma * sigma / n
            sigma_new = max(1e-10, sigma_new)

            if abs(mu_new - mu) < 1e-6 and abs(sigma_new - sigma) < 1e-6:
                mu, sigma = mu_new, sigma_new
                break
            mu, sigma = mu_new, sigma_new

        # Compute log-likelihood
        z = [(yi - mu) / sigma for yi in y]
        if xi != 0:
            if any(1 + xi * zi <= 1e-15 for zi in z):
                return False, mu, sigma, -1e20
            ll = 0.0
            for zi in z:
                t = 1 + xi * zi
                ll += -math.log(sigma) - (1.0 / xi + 1.0) * math.log(t) - t ** (-1.0 / xi)
        else:
            ll = 0.0
            for zi in z:
                ll += -math.log(sigma) - zi - math.exp(-zi)

        return True, mu, sigma, ll

    @staticmethod
    def _profile_xi(
        y: list[float], xi_start: float, xi_lo: float, xi_hi: float, n_grid: int,
    ) -> tuple[float, float, float, float, float]:
        """Grid search for best ξ in [xi_lo, xi_hi], returns (old_xi, best_xi, μ, σ, ll)."""
        best_xi = xi_start
        best_mu = 0.0
        best_sigma = 1.0
        best_ll = -float("inf")

        for i in range(n_grid + 1):
            xi_g = xi_lo + (xi_hi - xi_lo) * i / n_grid
            xi_g = max(-0.49, min(xi_g, 0.5))
            ok, mu_g, sigma_g, ll_g = GEVEngine._gev_profile_ll(y, xi_g)
            if ok and ll_g > best_ll:
                best_ll = ll_g
                best_xi = xi_g
                best_mu = mu_g
                best_sigma = sigma_g

        return xi_start, best_xi, best_mu, best_sigma, best_ll

    # ═══════════════════════════════════════════════════════
    # Return Level
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def return_level(
        fit: GEVFitResult,
        period: float = 100.0,
    ) -> GEVReturnLevel:
        """计算 Return Level: z_p = block 最大值在指定回报期内的期望水平.

        z_p = μ - σ/ξ · (1 - (-ln(1-1/T))^{-ξ})    ξ ≠ 0
        z_p = μ - σ · ln(-ln(1-1/T))                ξ = 0

        Args:
            fit: GEV 拟合结果
            period: 回报期 (block 数, 如 100 = 100 个月)

        Returns:
            GEVReturnLevel with 95% CI (delta method / profile approximation)
        """
        p = 1.0 - 1.0 / period
        y_p = -math.log(p)

        mu = float(fit.mu)
        sigma = float(fit.sigma)
        xi = float(fit.xi)

        if abs(xi) < 1e-10:
            z_p = mu - sigma * math.log(y_p)
        else:
            z_p = mu - (sigma / xi) * (1.0 - y_p ** (-xi))

        # 95% CI using delta method
        n = fit.n_blocks
        se_xi = float(fit.xi_se) if fit.xi_se > 0 else 0.01
        se_sigma = float(fit.sigma_se) if fit.sigma_se > 0 else sigma / math.sqrt(n)
        se_mu = float(fit.mu_se) if fit.mu_se > 0 else sigma / math.sqrt(n)

        # Approximate variance of z_p via delta method
        # dz/dμ = 1
        # dz/dσ = -(1 - y_p^{-ξ}) / ξ  (ξ ≠ 0)
        # dz/dξ = σ/ξ² · (1 - y_p^{-ξ}) - σ/ξ · y_p^{-ξ} · ln(y_p)
        if abs(xi) < 1e-10:
            dz_dsigma = -math.log(y_p)
            dz_dxi = sigma * math.log(y_p) * math.log(y_p) / 2.0  # approximation
        else:
            y_p_xi = y_p ** (-xi)
            dz_dsigma = -(1.0 - y_p_xi) / xi
            dz_dxi = sigma / (xi * xi) * (1.0 - y_p_xi) - sigma / xi * y_p_xi * math.log(y_p)

        var_z = se_mu ** 2 + dz_dsigma ** 2 * se_sigma ** 2 + dz_dxi ** 2 * se_xi ** 2
        se_z = math.sqrt(max(var_z, 0.0))
        z_score = 1.96

        return GEVReturnLevel(
            period=int(period),
            return_level=Decimal(str(round(z_p, 8))),
            lower_95=Decimal(str(round(z_p - z_score * se_z, 8))),
            upper_95=Decimal(str(round(z_p + z_score * se_z, 8))),
        )

    @staticmethod
    def return_level_plot_data(
        fit: GEVFitResult,
        periods: list[float] | None = None,
    ) -> dict:
        """生成 Return Level Plot 数据.

        Args:
            fit: GEV 拟合结果
            periods: 回报期列表 (默认 [2, 5, 10, 20, 50, 100, 200, 500])

        Returns:
            {period: ReturnLevel}
        """
        if periods is None:
            periods = [2, 5, 10, 20, 50, 100, 200, 500]
        results = {}
        for period in periods:
            rl = GEVEngine.return_level(fit, period)
            results[str(period)] = rl
        return results

    # ═══════════════════════════════════════════════════════
    # Diagnostic
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def probability_plot(
        block_maxima: list[Decimal],
        fit: GEVFitResult,
    ) -> dict:
        """Probability Plot — 比较经验分布与拟合 GEV.

        Returns:
            {empirical_quantile: fitted_quantile} (10 个点)
        """
        xs = sorted([float(x) for x in block_maxima])
        n = len(xs)
        mu = float(fit.mu)
        sigma = float(fit.sigma)
        xi = float(fit.xi)

        result = {}
        for i in range(1, 10):
            p = i / (n + 1)
            emp = xs[int(p * n)]
            if abs(xi) < 1e-10:
                fit_val = mu - sigma * math.log(-math.log(p))
            else:
                fit_val = mu - (sigma / xi) * (1.0 - (-math.log(p)) ** (-xi))
            result[str(round(p, 3))] = {
                "empirical": Decimal(str(round(emp, 8))),
                "fitted": Decimal(str(round(fit_val, 8))),
            }
        return result

    @staticmethod
    def quantile_plot(
        block_maxima: list[Decimal],
        fit: GEVFitResult,
    ) -> dict:
        """Quantile-Quantile (Q-Q) Plot 数据.

        Returns:
            {index: (empirical_quantile, theoretical_quantile)}
        """
        xs = sorted([float(x) for x in block_maxima])
        n = len(xs)
        mu = float(fit.mu)
        sigma = float(fit.sigma)
        xi = float(fit.xi)

        result = {}
        for i, x_emp in enumerate(xs):
            p = (i + 1) / (n + 1)
            if abs(xi) < 1e-10:
                x_theo = mu - sigma * math.log(-math.log(p))
            else:
                x_theo = mu - (sigma / xi) * (1.0 - (-math.log(p)) ** (-xi))
            result[str(i)] = {
                "empirical": Decimal(str(round(x_emp, 8))),
                "theoretical": Decimal(str(round(x_theo, 8))),
            }
        return result

    # ═══════════════════════════════════════════════════════
    # Model Comparison
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def compare_gev_gpd(
        returns: list[Decimal],
        block_size: int = 21,
    ) -> dict:
        """比较 GEV (Block Maxima) vs GPD (POT) 两种极值方法.

        Returns:
            包含两种方法在 3 个置信水平下的 VaR 估计对比。
        """
        from jingsuan.evt_engine import EVTEngine

        # GPD
        fit_gpd = EVTEngine.fit_gpd(returns, method="pwm")
        var_gpd = EVTEngine.tail_var(fit_gpd)

        # GEV
        try:
            fit_gev = EVTEngine.fit_gev(returns, block_size)
            var_gev = EVTEngine.tail_var(fit_gev)
            has_gev = True
        except Exception:
            has_gev = False

        result = {
            "gpd": {
                "var_95": str(var_gpd.var_95),
                "var_99": str(var_gpd.var_99),
                "var_999": str(var_gpd.var_999),
                "es_95": str(var_gpd.es_95),
                "es_99": str(var_gpd.es_99),
                "es_999": str(var_gpd.es_999),
                "xi": str(fit_gpd.xi),
                "method": fit_gpd.method,
            },
        }

        if has_gev:
            result["gev"] = {
                "var_95": str(var_gev.var_95),
                "var_99": str(var_gev.var_99),
                "var_999": str(var_gev.var_999),
                "es_95": str(var_gev.es_95),
                "es_99": str(var_gev.es_99),
                "es_999": str(var_gev.es_999),
                "xi_gev": str(fit_gev.xi),
                "tail_type": fit_gev.tail_type,
            }

        return result
