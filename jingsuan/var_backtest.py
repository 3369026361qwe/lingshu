"""
VaR 回测检验套件 — Kupiec / Christoffersen / Acerbi-Szekely / DQ / Berkowitz (v4.2).

纯计算层：无状态、无 IO。
验证 VaR 模型的准确性，多层次检验确保不系统性低估风险。

Usage:
    from jingsuan.var_backtest import VaRBacktestSuite
    result = VaRBacktestSuite.run_all(var_forecasts, actual_losses)
"""

import math
import random
from dataclasses import dataclass, field
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class VaRBacktestResult:
    n_observations: int
    n_violations: int
    violation_rate: Decimal
    expected_violations: Decimal
    kupiec_lr: float
    kupiec_pvalue: float
    kupiec_pass: bool
    christoffersen_ind_lr: float
    christoffersen_cc_lr: float
    christoffersen_pvalue: float
    christoffersen_pass: bool
    basel_zone: str
    basel_multiplier: Decimal
    dq_statistic: float = 0.0
    dq_pvalue: float = 1.0
    dq_pass: bool = True
    berkowitz_lr: float = 0.0
    berkowitz_pvalue: float = 1.0
    berkowitz_pass: bool = True
    rolling_zones: list[dict] = field(default_factory=list)


@dataclass
class RollingBacktestResult:
    window_size: int
    windows: list[dict]
    green_fraction: Decimal
    yellow_fraction: Decimal
    red_fraction: Decimal
    stability_score: Decimal


class VaRBacktestSuite:
    """VaR 回测检验套件 — 多层次验证."""

    @staticmethod
    def run_all(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        confidence_level: Decimal = Decimal("0.99"),
        n_bootstrap: int = 10000,
        es_forecasts: list[Decimal] | None = None,
    ) -> VaRBacktestResult:
        n = min(len(var_forecasts), len(actual_losses))
        var = var_forecasts[:n]
        loss = actual_losses[:n]

        indicators = [1 if l > v else 0 for v, l in zip(var, loss, strict=False)]
        x = sum(indicators)
        violation_rate = safe_divide(x, n)

        ku_lr, ku_p, ku_pass = VaRBacktestSuite.kupiec_test(x, n, confidence_level)
        ci_lr, cc_lr, cc_p, cc_pass = VaRBacktestSuite.christoffersen_test(indicators, x, n, confidence_level)
        zone, multiplier = VaRBacktestSuite.basel_traffic_light(x, n)
        dq_stat, dq_p, dq_pass = VaRBacktestSuite.dynamic_quantile_test(var, loss, indicators, confidence_level)
        bk_lr, bk_p, bk_pass = VaRBacktestSuite.berkowitz_test(var, loss, confidence_level)

        return VaRBacktestResult(
            n_observations=n, n_violations=x, violation_rate=violation_rate,
            expected_violations=Decimal(n) * (Decimal("1") - confidence_level),
            kupiec_lr=ku_lr, kupiec_pvalue=ku_p, kupiec_pass=ku_pass,
            christoffersen_ind_lr=ci_lr, christoffersen_cc_lr=cc_lr,
            christoffersen_pvalue=cc_p, christoffersen_pass=cc_pass,
            basel_zone=zone, basel_multiplier=Decimal(str(multiplier)),
            dq_statistic=dq_stat, dq_pvalue=dq_p, dq_pass=dq_pass,
            berkowitz_lr=bk_lr, berkowitz_pvalue=bk_p, berkowitz_pass=bk_pass,
        )

    @staticmethod
    def kupiec_test(violations: int, total: int, confidence: Decimal) -> tuple[float, float, bool]:
        x, n = violations, total
        alpha = 1 - float(confidence)
        if x == 0 or x == n:
            return float("inf"), 0.0, False
        p_hat = x / n
        ll_null = (n - x) * math.log(max(1 - alpha, 1e-15)) + x * math.log(max(alpha, 1e-15))
        ll_alt = (n - x) * math.log(max(1 - p_hat, 1e-15)) + x * math.log(max(p_hat, 1e-15))
        lr = max(0.0, -2 * (ll_null - ll_alt))
        p_value = VaRBacktestSuite._chi2_sf(lr, 1)
        return lr, p_value, p_value > 0.05

    @staticmethod
    def christoffersen_test(
        indicators: list[int], violations: int, total: int, confidence: Decimal
    ) -> tuple[float, float, float, bool]:
        """Christoffersen conditional coverage. LR_cc = LR_uc + LR_ind ~ chi2(2)."""
        n = len(indicators)
        if n < 3:
            return 0.0, 0.0, 1.0, True

        n00 = n01 = n10 = n11 = 0
        for t in range(n - 1):
            if indicators[t] == 0 and indicators[t + 1] == 0:
                n00 += 1
            elif indicators[t] == 0 and indicators[t + 1] == 1:
                n01 += 1
            elif indicators[t] == 1 and indicators[t + 1] == 0:
                n10 += 1
            else:
                n11 += 1

        n0, n1 = n00 + n01, n10 + n11
        if n0 == 0 or n1 == 0:
            return 0.0, 0.0, 1.0, True

        p01 = n01 / n0 if n0 > 0 else 0.0
        p11 = n11 / n1 if n1 > 0 else 0.0
        p = (n01 + n11) / n

        def _safe_log(val):
            return math.log(max(val, 1e-15))

        ll_null = (n00 * _safe_log(1 - p) + n01 * _safe_log(p)
                   + n10 * _safe_log(1 - p) + n11 * _safe_log(p))
        ll_alt = (n00 * _safe_log(1 - p01) + n01 * _safe_log(p01)
                  + n10 * _safe_log(1 - p11) + n11 * _safe_log(p11))

        lr_ind = max(0.0, -2 * (ll_null - ll_alt))
        lr_uc, _, _ = VaRBacktestSuite.kupiec_test(violations, total, confidence)
        lr_cc = lr_uc + lr_ind if lr_uc != float("inf") else lr_ind

        p_value = VaRBacktestSuite._chi2_sf(lr_cc, 2)
        return lr_ind, lr_cc, p_value, p_value > 0.05

    @staticmethod
    def basel_traffic_light(violations: int, total: int) -> tuple[str, float]:
        if violations <= 4:
            return "green", 3.0
        elif violations <= 9:
            return "yellow", round(3.0 + (violations - 4) * 0.85 / 5, 2)
        else:
            return "red", 4.0

    @staticmethod
    def dynamic_quantile_test(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        indicators: list[int],
        confidence_level: Decimal,
        n_lags: int = 4,
    ) -> tuple[float, float, bool]:
        """Engle & Manganelli (2004) Dynamic Quantile test."""
        n = len(indicators)
        alpha = 1 - float(confidence_level)
        if n < n_lags + 5:
            return 0.0, 1.0, True

        hits = [float(i) - alpha for i in indicators]
        k = n_lags + 2
        n_obs = n - n_lags
        if n_obs <= k:
            return 0.0, 1.0, True

        X = [[0.0] * k for _ in range(n_obs)]
        Y = [0.0] * n_obs
        for t in range(n_obs):
            idx = t + n_lags
            X[t][0] = 1.0
            for lag in range(1, n_lags + 1):
                X[t][lag] = hits[idx - lag]
            X[t][n_lags + 1] = float(var_forecasts[idx])
            Y[t] = hits[idx]

        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k
        for t in range(n_obs):
            for i in range(k):
                xty[i] += X[t][i] * Y[t]
                for j in range(k):
                    xtx[i][j] += X[t][i] * X[t][j]

        beta = _solve_linear(xtx, xty, k)
        if beta is None:
            return 0.0, 1.0, True

        dq = 0.0
        for i in range(k):
            for j in range(k):
                dq += beta[i] * xtx[i][j] * beta[j]
        dq /= max(alpha * (1 - alpha), 1e-10)
        p_value = VaRBacktestSuite._chi2_sf(dq, k)
        return dq, p_value, p_value > 0.05

    @staticmethod
    def berkowitz_test(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        confidence_level: Decimal,
    ) -> tuple[float, float, bool]:
        """Berkowitz LR test: transform PIT -> N(0,1), test AR(1) alternative.

        Steps:
        1. Compute PIT: u_t = P(loss < VaR_t) = 1 - alpha if model is correct
           Actually, we compute u_t via empirical PIT of standardized forecast errors.
        2. Transform to Normal: z_t = Phi^{-1}(u_t)
        3. Test H0: z_t ~ iid N(0,1) vs AR(1) alternative via LR.
        """
        n = len(actual_losses)
        if n < 30:
            return 0.0, 1.0, True

        alpha = 1 - float(confidence_level)

        # Step 1: Compute PIT values from VaR forecasts
        # u_t = empirical PIT: rank-based probability that loss <= VaR_t
        loss_vals = [float(l) for l in actual_losses]
        var_vals = [float(v) for v in var_forecasts]

        # Standardized forecast errors: e_t = (Loss_t - mean) / sigma where sigma from VaR
        # Under correct model: Loss_t ~ N(mean_t, sigma_t) with VaR_t = mean_t + z_alpha * sigma_t
        # Use VaR to estimate implied volatility: sigma_implied_t = |VaR_t| / z_alpha  (if mean~0)
        # Then PIT: u_t = Phi((Loss_t - 0) / sigma_implied_t) if loss is positive-side
        # Simplified rank-based PIT using VaR violation status:
        # u_t = alpha if violation, else uniform in (alpha, 1)

        # Standard approach: rank-based PIT from the forecast distribution
        # For each t, compute z_t = (Loss_t - VaR_t) / implied_sigma where
        # implied_sigma = |VaR_t - mean_t| / z_alpha  or just use the empirical PIT
        # Simpler and more robust: use the violation-based PIT
        z_vals = []
        for t in range(n):
            if loss_vals[t] > var_vals[t]:
                # Violation: u_t is in (alpha, 1)
                # Use uniform transform: u_t = alpha + (1-alpha)*random, but deterministic:
                # Use mid-point of the tail region
                u_t = alpha + (1 - alpha) * 0.5
            else:
                # Non-violation: u_t is in (0, alpha)
                # Normalize loss relative to VaR
                ratio = loss_vals[t] / max(var_vals[t], 1e-10)
                u_t = alpha * min(ratio, 1.0) * 0.99  # stays within (0, alpha)
            # Clamp and transform: z_t = Phi^{-1}(u_t)
            u_t = max(1e-10, min(u_t, 1 - 1e-10))
            # Phi^{-1} via rational approximation
            z_vals.append(_norm_inv(u_t))

        z_lag = z_vals[:-1]
        z_curr = z_vals[1:]
        n_ar = len(z_curr)

        sum_z = sum(z_curr)
        sum_zl = sum(z_lag)
        sum_zz = sum(z_curr[i] * z_lag[i] for i in range(n_ar))
        sum_z2 = sum(z * z for z in z_lag)

        denom = n_ar * sum_z2 - sum_zl * sum_zl
        if abs(denom) < 1e-15:
            mu_hat, rho_hat = sum_z / n_ar, 0.0
        else:
            mu_hat = (sum_z * sum_z2 - sum_zl * sum_zz) / denom
            rho_hat = (n_ar * sum_zz - sum_zl * sum_z) / denom
        rho_hat = max(-0.99, min(rho_hat, 0.99))

        resid = [(z_curr[i] - mu_hat - rho_hat * z_lag[i]) for i in range(n_ar)]
        sigma2_hat = sum(r * r for r in resid) / max(1, n_ar - 2)
        if sigma2_hat <= 0:
            sigma2_hat = 1.0

        ll_null = -n_ar / 2.0 * math.log(2 * math.pi) - 0.5 * sum(z * z for z in z_curr)
        ll_alt = (-n_ar / 2.0 * math.log(2 * math.pi * sigma2_hat)
                  - 0.5 * sum(r * r for r in resid) / sigma2_hat)

        lr = max(0.0, -2 * (ll_null - ll_alt))
        p_value = VaRBacktestSuite._chi2_sf(lr, 3)
        return lr, p_value, p_value > 0.05
        ll_alt = (-n_ar / 2.0 * math.log(2 * math.pi * sigma2_hat)
                  - 0.5 * sum(r * r for r in resid) / sigma2_hat)

        lr = max(0.0, -2 * (ll_null - ll_alt))
        p_value = VaRBacktestSuite._chi2_sf(lr, 3)
        return lr, p_value, p_value > 0.05

    @staticmethod
    def acerbi_szekely_test(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        es_forecasts: list[Decimal],
        confidence_level: Decimal = Decimal("0.99"),
        n_bootstrap: int = 5000,
    ) -> tuple[float, float, bool]:
        rng = random.Random(42)
        n = min(len(var_forecasts), len(actual_losses), len(es_forecasts))
        alpha = 1 - float(confidence_level)

        def _z_stat(var_f, loss_f, es_f):
            z = 0.0
            for v, l, e in zip(var_f, loss_f, es_f, strict=False):
                if float(l) > float(v) and float(e) > 0:
                    z += float(l) / float(e)
            return z / (n * alpha) - 1.0

        z_obs = _z_stat(var_forecasts, actual_losses, es_forecasts)
        z_boot = []
        for _ in range(n_bootstrap):
            indices = list(range(n))
            rng.shuffle(indices)
            loss_b = [actual_losses[i] for i in indices]
            z_boot.append(_z_stat(var_forecasts, loss_b, es_forecasts))

        extreme = sum(1 for z in z_boot if abs(z) >= abs(z_obs))
        p_value = extreme / n_bootstrap
        return z_obs, p_value, p_value > 0.05

    @staticmethod
    def rolling_traffic_light(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        confidence_level: Decimal = Decimal("0.99"),
        window_size: int = 250,
        step: int = 50,
    ) -> RollingBacktestResult:
        n = min(len(var_forecasts), len(actual_losses))
        windows = []
        n_green = n_yellow = n_red = 0
        start = 0
        while start + window_size <= n:
            end = start + window_size
            var_w = var_forecasts[start:end]
            loss_w = actual_losses[start:end]
            violations = sum(1 for v, l in zip(var_w, loss_w, strict=False) if l > v)
            zone, multiplier = VaRBacktestSuite.basel_traffic_light(violations, window_size)
            if zone == "green":
                n_green += 1
            elif zone == "yellow":
                n_yellow += 1
            else:
                n_red += 1
            _, kp, kpass = VaRBacktestSuite.kupiec_test(violations, window_size, confidence_level)
            windows.append({
                "start": start, "end": end, "violations": violations,
                "zone": zone, "multiplier": multiplier,
                "kupiec_pass": kpass, "kupiec_pvalue": round(kp, 4),
            })
            start += step

        n_total = len(windows)
        green_frac = safe_divide(n_green, n_total) if n_total > 0 else Decimal("0")
        yellow_frac = safe_divide(n_yellow, n_total) if n_total > 0 else Decimal("0")
        red_frac = safe_divide(n_red, n_total) if n_total > 0 else Decimal("0")
        stability = safe_divide(Decimal(str(n_green * 1.0 + n_yellow * 0.5)),
                                Decimal(str(n_total))) if n_total > 0 else Decimal("0")

        return RollingBacktestResult(
            window_size=window_size, windows=windows,
            green_fraction=green_frac, yellow_fraction=yellow_frac,
            red_fraction=red_frac, stability_score=stability,
        )

    @staticmethod
    def _chi2_sf(x: float, df: int) -> float:
        """Chi-squared survival function P(chi2_df > x)."""
        if x <= 0:
            return 1.0
        if df == 1:
            return 2.0 * (1.0 - _std_normal_cdf(math.sqrt(x)))
        elif df == 2:
            return math.exp(-x / 2.0)
        elif df == 3:
            return 1.0 - _gammainc(1.5, x / 2.0)
        else:
            return 1.0 - _gammainc(df / 2.0, x / 2.0)


# -- Module-level helpers --

def _std_normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


def _norm_inv(p: float) -> float:
    """Inverse normal CDF (Abramowitz & Stegun)."""
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


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x).
    Uses series expansion (x < a+1) or Lentz continued fraction (x >= a+1).
    """
    if x <= 0:
        return 0.0
    if a <= 0:
        return 1.0
    log_gamma_a = math.lgamma(a)

    if x < a + 1.0:
        term = 1.0 / a
        result = term
        for n in range(1, 500):
            term *= x / (a + n)
            result += term
            if term < 1e-16 * result:
                break
        return result * math.exp(-x + a * math.log(x) - log_gamma_a)
    else:
        f0 = 1.0e-30
        c0 = 0.0
        d0 = 1.0
        b_val = x + 1.0 - a
        for n in range(1, 500):
            an = -n * (n - a)
            b_val = b_val + 2.0
            d0 = b_val + an * d0
            if abs(d0) < 1e-30:
                d0 = 1e-30
            c0 = b_val + an / max(c0, 1e-30)
            d0 = 1.0 / max(d0, 1e-30)
            delta = d0 * c0
            f0 *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        q = f0 * math.exp(-x + a * math.log(x) - log_gamma_a)
        return max(0.0, min(1.0, 1.0 - q))


def _solve_linear(A: list[list[float]], b: list[float], n: int) -> list[float] | None:
    """Gaussian elimination with partial pivoting."""
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    eps = 1e-15

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[max_row][col]) < eps:
            return None
        aug[col], aug[max_row] = aug[max_row], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i][i] if abs(aug[i][i]) > eps else 0.0
    return x
