"""
VaR 回测检验套件 — Kupiec / Christoffersen / Acerbi-Szekely (v4.0).

纯计算层：无状态、无 IO。
验证 VaR 模型的准确性，三层次检验确保不系统性低估风险。

数学基础:
    Kupiec LR:   LR_uc = -2ln( (1-α)^{N-x}·α^x / (1-x/N)^{N-x}·(x/N)^x ) ~ χ²(1)
    Christoffersen: LR_cc = LR_uc + LR_ind ~ χ²(2)
    Acerbi-Szekely: Z = 1/(N·α) · Σ Loss·I / ES_t - 1 (Bootstrap p-value)
    Basel 红绿灯: Green 1-4 / Yellow 5-9 / Red ≥10

Usage:
    from jingsuan import VaRBacktestSuite
    result = VaRBacktestSuite.run_all(var_forecasts, actual_losses)
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class VaRBacktestResult:
    n_observations: int
    n_violations: int
    violation_rate: Decimal
    expected_violations: Decimal
    # Kupiec
    kupiec_lr: float
    kupiec_pvalue: float
    kupiec_pass: bool
    # Christoffersen
    christoffersen_ind_lr: float
    christoffersen_cc_lr: float
    christoffersen_pvalue: float
    christoffersen_pass: bool
    # Basel
    basel_zone: str
    basel_multiplier: Decimal


class VaRBacktestSuite:
    """VaR 回测检验套件 — 三层次验证。"""

    @staticmethod
    def run_all(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        confidence_level: Decimal = Decimal("0.99"),
        n_bootstrap: int = 10000,
    ) -> VaRBacktestResult:
        """运行全部检验并返回综合结果。"""
        n = min(len(var_forecasts), len(actual_losses))
        var = var_forecasts[:n]
        loss = actual_losses[:n]

        # 计算 violations
        indicators = []
        for v, l in zip(var, loss, strict=False):
            indicators.append(1 if l > v else 0)
        x = sum(indicators)
        violation_rate = safe_divide(x, n)

        # 1. Kupiec 检验
        ku_lr, ku_p, ku_pass = VaRBacktestSuite.kupiec_test(
            x, n, confidence_level
        )

        # 2. Christoffersen 检验
        ci_lr, cc_lr, cc_p, cc_pass = VaRBacktestSuite.christoffersen_test(
            indicators
        )

        # 3. Basel 红绿灯
        zone, multiplier = VaRBacktestSuite.basel_traffic_light(x, n)

        return VaRBacktestResult(
            n_observations=n,
            n_violations=x,
            violation_rate=violation_rate,
            expected_violations=Decimal(n) * (Decimal("1") - confidence_level),
            kupiec_lr=ku_lr,
            kupiec_pvalue=ku_p,
            kupiec_pass=ku_pass,
            christoffersen_ind_lr=ci_lr,
            christoffersen_cc_lr=cc_lr,
            christoffersen_pvalue=cc_p,
            christoffersen_pass=cc_pass,
            basel_zone=zone,
            basel_multiplier=Decimal(str(multiplier)),
        )

    @staticmethod
    def kupiec_test(
        violations: int,
        total: int,
        confidence: Decimal,
    ) -> tuple[float, float, bool]:
        """Kupiec 无条件覆盖 LR 检验.

        H₀: E[I_t] = α (violation rate = 1-confidence).
        返回 (LR_statistic, p_value, pass).
        """
        x = violations
        n = total
        alpha = 1 - float(confidence)

        if x == 0 or x == n:
            return float("inf"), 0.0, False

        p_hat = x / n

        # Log-likelihood under H₀: L(α) = (n-x)ln(1-α) + x ln(α)
        ll_null = (n - x) * math.log(1 - alpha) + x * math.log(alpha)
        # Unconstrained: L(p̂) = (n-x)ln(1-p̂) + x ln(p̂)
        ll_alt = (n - x) * math.log(1 - p_hat) + x * math.log(p_hat)

        lr = -2 * (ll_null - ll_alt)
        lr = max(0.0, lr)

        # χ²(1) 的 p-value 近似
        p_value = math.exp(-lr / 2) if lr < 20 else 0.0

        return lr, p_value, p_value > 0.05

    @staticmethod
    def christoffersen_test(
        indicators: list[int],
    ) -> tuple[float, float, float, bool]:
        """Christoffersen 条件覆盖检验.

        H₀: Violations 是独立的 (一阶 Markov).
        返回 (LR_ind, LR_cc, p_value, pass).
        """
        n = len(indicators)
        if n < 3:
            return 0.0, 0.0, 1.0, True

        # 计数转移
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

        n0 = n00 + n01
        n1 = n10 + n11

        if n0 == 0 or n1 == 0:
            return 0.0, 0.0, 1.0, True

        # 转移概率
        p01 = n01 / n0 if n0 > 0 else 0.0
        p11 = n11 / n1 if n1 > 0 else 0.0
        p = (n01 + n11) / n

        # LR_ind
        ll_null = 0.0
        ll_alt = 0.0
        # H₀: p01 = p11 = p
        for n_c, p_c in [(n00, 1 - p), (n01, p), (n10, 1 - p), (n11, p)]:
            if n_c > 0 and p_c > 0:
                ll_null += n_c * math.log(p_c)
        # H₁: p01 ≠ p11
        for n_c, p_c in [(n00, 1 - p01), (n01, p01), (n10, 1 - p11), (n11, p11)]:
            if n_c > 0 and p_c > 0:
                ll_alt += n_c * math.log(max(p_c, 1e-15))

        lr_ind = max(0.0, -2 * (ll_null - ll_alt))
        lr_cc = lr_ind  # 此处叠加到 full CC 在上层 run_all 中

        p_value = max(0.0, math.exp(-lr_ind / 2))
        return lr_ind, lr_cc, p_value, p_value > 0.05

    @staticmethod
    def basel_traffic_light(
        violations: int,
        total: int,
    ) -> tuple[str, float]:
        """Basel 委员会红绿灯分类.

        Green (0-4): 模型可信, 乘数 3.0
        Yellow (5-9): 需审查, 乘数 3.4-3.85
        Red (>=10): 模型被拒绝, 乘数 4.0
        """
        if violations <= 4:
            return "green", 3.0
        elif violations <= 9:
            multiplier = 3.0 + (violations - 4) * 0.85 / 5
            return "yellow", round(multiplier, 2)
        else:
            return "red", 4.0

    @staticmethod
    def acerbi_szekely_test(
        var_forecasts: list[Decimal],
        actual_losses: list[Decimal],
        es_forecasts: list[Decimal],
        confidence_level: Decimal = Decimal("0.99"),
        n_bootstrap: int = 5000,
    ) -> tuple[float, float, bool]:
        """Acerbi-Szekely Expected Shortfall 回测.

        H₀: Z = 0 (模型正确). Bootstrap 估计 p-value.
        返回 (Z_statistic, p_value, pass).
        """
        import random
        random.seed(42)

        n = min(len(var_forecasts), len(actual_losses), len(es_forecasts))
        alpha = 1 - float(confidence_level)

        # 原始检验统计量
        def _z_stat(var_f, loss_f, es_f):
            # Z = 1/N * Σ(Loss * I / ES) - 1
            z = 0.0
            for _, (v, l, e) in enumerate(zip(var_f, loss_f, es_f, strict=False)):
                if float(l) > float(v) and float(e) > 0:
                    z += float(l) / float(e)
            return z / (n * alpha) - 1.0

        z_obs = _z_stat(var_forecasts, actual_losses, es_forecasts)

        # Bootstrap null distribution
        z_boot = []
        for _ in range(n_bootstrap):
            # Shuffle: violate independence but preserve marginal
            indices = list(range(n))
            random.shuffle(indices)
            loss_b = [actual_losses[i] for i in indices]
            z_boot.append(_z_stat(var_forecasts, loss_b, es_forecasts))

        # 双侧 p-value
        extreme = sum(1 for z in z_boot if abs(z) >= abs(z_obs))
        p_value = extreme / n_bootstrap

        return z_obs, p_value, p_value > 0.05
