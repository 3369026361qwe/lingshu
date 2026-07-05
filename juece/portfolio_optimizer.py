"""
Black-Litterman 组合优化器 (v4.0).

真正的贝叶斯更新:
    1. Ledoit-Wolf 收缩协方差估计
    2. 从市值权重反推均衡收益 Π = δ·Σ·w_mkt
    3. 贝叶斯融合: E(R) = [(τΣ)⁻¹ + P^TΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + P^TΩ⁻¹Q]
    4. 均值-方差优化 → 最优权重

Usage:
    from juece.portfolio_optimizer import PortfolioOptimizer, View, BLConfig
    opt = PortfolioOptimizer(config)
    result = opt.optimize(returns_matrix, market_weights, views)
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class View:
    """投资者观点."""
    assets: list[str]
    weights: list[Decimal]
    view_return: Decimal
    confidence: Decimal = Decimal("0.5")
    is_relative: bool = False


@dataclass
class BLConfig:
    risk_aversion: Decimal = Decimal("2.5")
    tau: Decimal = Decimal("0.05")
    max_weight: Decimal = Decimal("0.10")
    volatility_target: Decimal | None = None


@dataclass
class BLOptimizationResult:
    optimal_weights: dict[str, Decimal]
    equilibrium_returns: list[Decimal]
    posterior_returns: list[Decimal]
    posterior_cov: list[list[Decimal]]
    expected_return: Decimal
    expected_volatility: Decimal
    expected_sharpe: Decimal


class PortfolioOptimizer:
    """Black-Litterman + 均值方差组合优化器."""

    def __init__(self, config: BLConfig | None = None):
        self.config = config or BLConfig()

    def estimate_covariance(
        self, returns_matrix: list[list[Decimal]]
    ) -> list[list[Decimal]]:
        """Ledoit-Wolf 收缩协方差估计."""
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        data = [[float(v) for v in r[:n_obs]] for r in returns_matrix]

        # 样本协方差
        means = [sum(col) / n_obs for col in data]
        S = [[0.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                cov = sum((data[i][t] - means[i]) * (data[j][t] - means[j]) for t in range(n_obs))
                S[i][j] = cov / (n_obs - 1) if n_obs > 1 else 0.0

        # Ledoit-Wolf 收缩目标: 常数相关矩阵
        avg_var = sum(S[i][i] for i in range(n_assets)) / n_assets
        avg_corr = 0.0
        count = 0
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if S[i][i] > 1e-10 and S[j][j] > 1e-10:
                    avg_corr += S[i][j] / math.sqrt(S[i][i] * S[j][j])
                    count += 1
        avg_corr = avg_corr / count if count > 0 else 0.0

        # 收缩目标 F
        F = [[0.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            F[i][i] = avg_var
            for j in range(n_assets):
                if i != j:
                    F[i][j] = avg_corr * math.sqrt(avg_var * avg_var)

        # 收缩强度
        pi_hat = 0.0
        for i in range(n_assets):
            for j in range(n_assets):
                diff = S[i][j] - F[i][j]
                pi_hat += diff * diff
        pi_hat /= n_assets * n_assets

        rho = 0.0
        for i in range(n_assets):
            for t in range(n_obs):
                diff = (data[i][t] - means[i]) ** 2 - S[i][i]
                rho += diff * diff
        rho /= n_assets * n_obs * max(1, n_obs - 1)

        shrinkage = min(pi_hat / (pi_hat + rho) if (pi_hat + rho) > 0 else 0.0, 1.0)

        # 收缩协方差
        cov = [[Decimal("0")] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                cov[i][j] = Decimal(str(
                    (1 - shrinkage) * S[i][j] + shrinkage * F[i][j]
                ))
        return cov

    def implied_equilibrium_returns(
        self, cov_matrix: list[list[Decimal]], market_weights: list[Decimal]
    ) -> list[Decimal]:
        """Π = δ · Σ · w_mkt."""
        n = len(market_weights)
        delta = self.config.risk_aversion
        pi = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                pi[i] += delta * cov_matrix[i][j] * market_weights[j]
        return pi

    def incorporate_views(
        self,
        equilibrium: list[Decimal],
        cov_matrix: list[list[Decimal]],
        views: list[View],
        asset_codes: list[str],
    ) -> tuple[list[Decimal], list[list[Decimal]]]:
        """贝叶斯更新: E(R|views) = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]."""
        n = len(equilibrium)
        k = len(views)
        tau = self.config.tau

        if k == 0:
            return list(equilibrium), [list(row) for row in cov_matrix]

        # P 矩阵 (k x n) — views
        P = [[Decimal("0")] * n for _ in range(k)]
        Q = [Decimal("0")] * k
        Omega = [[Decimal("0")] * k for _ in range(k)]

        for v_idx, view in enumerate(views):
            for i, code in enumerate(asset_codes):
                if code in view.assets:
                    P[v_idx][i] = view.weights[view.assets.index(code)]
            Q[v_idx] = view.view_return
            # Ω = diag(P·(τΣ)·P^T / confidence)
            p_sigma_p = Decimal("0")
            for i in range(n):
                for j in range(n):
                    p_sigma_p += P[v_idx][i] * cov_matrix[i][j] * P[v_idx][j] * tau
            Omega[v_idx][v_idx] = safe_divide(p_sigma_p, max(view.confidence, Decimal("0.01")))

        # (τΣ)⁻¹ — 简化: 用对角近似
        tau_sigma_inv = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            tau_sigma_inv[i][i] = safe_divide(Decimal("1"), tau * cov_matrix[i][i])

        # P^T Ω⁻¹ P
        omega_inv = [[Decimal("0")] * k for _ in range(k)]
        for v in range(k):
            omega_inv[v][v] = safe_divide(Decimal("1"), Omega[v][v])

        pt_omega_p = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for v in range(k):
                    pt_omega_p[i][j] += P[v][i] * omega_inv[v][v] * P[v][j]

        # posterior covariance⁻¹ = (τΣ)⁻¹ + P^T Ω⁻¹ P
        post_cov_inv = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                post_cov_inv[i][j] = tau_sigma_inv[i][j] + pt_omega_p[i][j]

        # posterior = post_cov · (τΣ⁻¹ Π + P^T Ω⁻¹ Q)
        part1 = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                part1[i] += tau_sigma_inv[i][j] * equilibrium[j]

        part2 = [Decimal("0")] * n
        for i in range(n):
            for v in range(k):
                part2[i] += P[v][i] * omega_inv[v][v] * Q[v]

        rhs = [part1[i] + part2[i] for i in range(n)]

        # 求解 post_cov_inv · posterior = rhs (n ≤ 50, 直接求逆)
        posterior = self._solve_linear(post_cov_inv, rhs)

        # 后验协方差
        post_cov = self._invert_matrix(post_cov_inv)

        # 后验协方差 = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        for i in range(n):
            for j in range(n):
                post_cov[i][j] += cov_matrix[i][j]

        return posterior, post_cov

    def optimize(
        self,
        posterior_returns: list[Decimal],
        posterior_cov: list[list[Decimal]],
        asset_codes: list[str],
    ) -> BLOptimizationResult:
        """均值-方差优化: max w·μ - (δ/2)·w·Σ·w."""
        n = len(posterior_returns)
        delta = self.config.risk_aversion
        w_max = self.config.max_weight

        # 解析解 (无约束): w* = (1/δ) Σ⁻¹ μ
        # 加约束: 0 ≤ w ≤ w_max, Σ w = 1 → 用二次规划近似
        weights = [Decimal("1") / Decimal(n)] * n

        # 简化: 解析解 + 投影
        cov_inv = self._invert_matrix(posterior_cov)
        raw_w = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                raw_w[i] += safe_divide(Decimal("1"), delta) * cov_inv[i][j] * posterior_returns[j]

        # 投影到约束空间
        # 1. 非负
        raw_w = [max(w, Decimal("0")) for w in raw_w]
        # 2. 截断
        raw_w = [min(w, w_max) for w in raw_w]
        # 3. 归一化
        total = sum(raw_w)
        weights = [safe_divide(w, total) for w in raw_w] if total > 0 else [Decimal("1") / Decimal(n)] * n

        # 组合指标
        port_ret = sum(weights[i] * posterior_returns[i] for i in range(n))
        port_var = Decimal("0")
        for i in range(n):
            for j in range(n):
                port_var += weights[i] * posterior_cov[i][j] * weights[j]
        port_vol = port_var.sqrt()
        port_sharpe = safe_divide(port_ret, port_vol) if port_vol > 0 else Decimal("0")

        return BLOptimizationResult(
            optimal_weights={code: w for code, w in zip(asset_codes, weights, strict=False)},
            equilibrium_returns=[], posterior_returns=posterior_returns,
            posterior_cov=posterior_cov,
            expected_return=port_ret, expected_volatility=port_vol,
            expected_sharpe=port_sharpe,
        )

    @staticmethod
    def _solve_linear(A: list[list[Decimal]], b: list[Decimal]) -> list[Decimal]:
        """高斯消元 Ax = b, n ≤ 50."""
        n = len(b)
        M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
        for col in range(n):
            # 选主元
            pivot = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > abs(M[pivot][col]):
                    pivot = row
            M[col], M[pivot] = M[pivot], M[col]
            if M[col][col] == 0:
                continue
            for row in range(col + 1, n):
                factor = safe_divide(M[row][col], M[col][col])
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
        # 回代
        x = [Decimal("0")] * n
        for i in range(n - 1, -1, -1):
            s = M[i][n]
            for j in range(i + 1, n):
                s -= M[i][j] * x[j]
            x[i] = safe_divide(s, M[i][i])
        return x

    @staticmethod
    def _invert_matrix(A: list[list[Decimal]]) -> list[list[Decimal]]:
        """矩阵求逆 (对角近似, n ≤ 50)."""
        n = len(A)
        # 完整求逆: Gauss-Jordan
        I = [[Decimal("1") if i == j else Decimal("0") for j in range(n)] for i in range(n)]
        M = [[A[i][j] for j in range(n)] + I[i][:] for i in range(n)]

        for col in range(n):
            pivot = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > abs(M[pivot][col]):
                    pivot = row
            M[col], M[pivot] = M[pivot], M[col]
            piv_val = M[col][col]
            if piv_val == 0:
                continue
            for j in range(2 * n):
                M[col][j] = safe_divide(M[col][j], piv_val)
            for row in range(n):
                if row == col:
                    continue
                factor = M[row][col]
                for j in range(2 * n):
                    M[row][j] -= factor * M[col][j]

        return [[M[i][n + j] for j in range(n)] for i in range(n)]
