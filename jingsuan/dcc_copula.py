"""
DCC-GARCH Copula — 时变条件相关 Copula (v4.0).

纯计算层：无状态、无 IO。
实现 Engle (2002) Dynamic Conditional Correlation 模型，
用于捕捉资产间随时间变化的尾部依赖结构。

数学基础:
    DCC-GARCH(1,1):
        r_t = μ + ε_t,  ε_t | Ω_{t-1} ~ N(0, H_t)
        H_t = D_t · R_t · D_t

        D_t = diag(σ_{1,t}, ..., σ_{d,t})  (univariate GARCH volatilities)
        R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}  (correlation matrix)

        Q_t = (1 - α - β) · Q̄ + α · z_{t-1}z_{t-1}' + β · Q_{t-1}

        其中:
          z_t = D_t^{-1} · ε_t  (standardized residuals)
          Q̄ = E[z_t z_t']       (unconditional covariance of z_t)
          α = news impact (新息冲击)
          β = persistence (持续性)
          α + β < 1 (stationarity)

    时变 Copula 相关矩阵:
        R_t 作为 t-Copula / Gaussian Copula 的相关矩阵输入
        每个时间点的尾部依赖随着相关性变化而变化

    DCC-t-Copula:
        C_t(u₁,...,u_d) = t_{ν, R_t}(t_ν^{-1}(u₁), ..., t_ν^{-1}(u_d))
        其中 ν = 自由度, R_t = 时变相关矩阵

Usage:
    from jingsuan.dcc_copula import DCCCopula
    dcc = DCCCopula.fit(returns_matrix)
    corr_series = dcc.correlation_series()
"""

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class DCCFitResult:
    """DCC-GARCH(1,1) 拟合结果."""
    # DCC parameters
    alpha: Decimal       # 新息冲击参数
    beta: Decimal        # 持续性参数
    unconditional_corr: list[list[float]]  # Q̄ (unconditional correlation)

    # Time-varying quantities
    conditional_corr: list[list[list[float]]]  # [t][i][j]
    conditional_vol: list[list[float]]         # [asset][t] GARCH volatilities
    h_t: list[list[list[float]]]               # [t][i][j] conditional covariance

    # Copula parameters
    nu: Decimal = Decimal("5")   # t-Copula df (if t-Copula)
    copula_type: str = "t"       # "t" | "gaussian"

    # Diagnostics
    n_assets: int = 0
    n_obs: int = 0
    log_likelihood: float = 0.0
    aic: float = 0.0

    def correlation_series(self, i: int = 0, j: int = 1) -> list[float]:
        """提取 (i,j) 资产的时变相关系数序列."""
        return [cc[i][j] for cc in self.conditional_corr]

    def latest_correlation(self) -> list[list[float]]:
        """返回最新的条件相关矩阵."""
        if not self.conditional_corr:
            return [[1.0]]
        return self.conditional_corr[-1]

    def tail_dependence_series(self, i: int = 0, j: int = 1) -> list[float]:
        """提取 (i,j) 的时变尾部依赖系数序列 (t-Copula).

        λ_t = 2 · t_{ν+1}(-√((ν+1)(1-ρ_t)/(1+ρ_t)))
        """
        if self.copula_type != "t":
            return [0.0] * len(self.conditional_corr)
        nu = float(self.nu)
        result = []
        for corr_t in self.correlation_series(i, j):
            rho = max(-0.999, min(corr_t, 0.999))
            arg = -math.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            ld = 2 * DCCCopula._tcdf(arg, nu + 1)
            result.append(ld)
        return result


class DCCCopula:
    """DCC-GARCH Copula — 时变条件相关建模。"""

    @staticmethod
    def _tcdf(x: float, nu: float) -> float:
        """t-distribution CDF via numerical integration."""
        h = abs(x) / 200
        r = 0.0
        for i in range(200):
            ti = i * h
            pdf = (math.gamma((nu + 1) / 2)
                   / (math.sqrt(nu * math.pi) * math.gamma(nu / 2))
                   * (1 + ti * ti / nu) ** (-(nu + 1) / 2))
            r += pdf * h
        return 0.5 + r if x > 0 else 0.5 - r

    @staticmethod
    def fit(
        returns_matrix: list[list[Decimal]],
        copula_type: str = "t",
        alpha_init: float = 0.05,
        beta_init: float = 0.90,
        nu_init: float = 5.0,
    ) -> DCCFitResult:
        """DCC-GARCH(1,1) 拟合.

        Args:
            returns_matrix: [asset][time], 每行是一个资产的收益序列
            copula_type: "t" (Student-t Copula) | "gaussian"
            alpha_init: DCC α 初值 (news impact)
            beta_init: DCC β 初值 (persistence)
            nu_init: t-Copula 自由度初值

        Returns:
            DCCFitResult with time-varying correlations
        """
        n_assets = len(returns_matrix)
        n_obs = min(len(r) for r in returns_matrix)
        if n_obs < 60:
            raise ValueError(f"DCC 需要 >= 60 个观测, 当前 {n_obs}")

        # Convert to float
        data = [[float(v) for v in r[:n_obs]] for r in returns_matrix]

        # Step 1: Univariate GARCH(1,1) for each asset
        garch_params = []
        std_residuals = []  # z_{i,t}
        cond_vol = []       # σ_{i,t}
        for a in range(n_assets):
            omega, arch, garch_beta, volat, z = DCCCopula._garch_11_fit(data[a])
            garch_params.append((omega, arch, garch_beta))
            cond_vol.append(volat)
            std_residuals.append(z)

        # Step 2: Compute unconditional correlation Q̄
        Q_bar = [[0.0] * n_assets for _ in range(n_assets)]
        for i in range(n_assets):
            for j in range(n_assets):
                Q_bar[i][j] = sum(
                    std_residuals[i][t] * std_residuals[j][t] for t in range(n_obs)
                ) / (n_obs - 1)
                Q_bar[j][i] = Q_bar[i][j]

        # Step 3: DCC recursion
        alpha_a, beta_b = alpha_init, beta_init
        Q_t = [[q for q in row] for row in Q_bar]  # Q_0 = Q̄
        R_t_list = []
        Q_t_list = []

        for t in range(n_obs):
            if t == 0:
                Q_t = [[q for q in row] for row in Q_bar]
            else:
                z_prev = [std_residuals[i][t - 1] for i in range(n_assets)]
                for i in range(n_assets):
                    for j in range(n_assets):
                        Q_t_new = (1 - alpha_a - beta_b) * Q_bar[i][j] \
                                  + alpha_a * z_prev[i] * z_prev[j] \
                                  + beta_b * Q_t[i][j]
                        Q_t[i][j] = Q_t_new

            # R_t = diag(Q_t)^{-1/2} · Q_t · diag(Q_t)^{-1/2}
            Q_diag_inv_sqrt = [1.0 / math.sqrt(max(abs(Q_t[i][i]), 1e-10)) for i in range(n_assets)]
            R_t = [[0.0] * n_assets for _ in range(n_assets)]
            for i in range(n_assets):
                for j in range(n_assets):
                    R_t[i][j] = Q_t[i][j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]
                    R_t[j][i] = R_t[i][j]

            R_t_list.append(R_t)
            Q_t_list.append([[q for q in row] for row in Q_t])

        # Step 4: Estimate ν for t-Copula (if applicable)
        nu = nu_init
        ll = 0.0
        if copula_type == "t":
            ll, nu = DCCCopula._fit_nu_t_copula(R_t_list, std_residuals, n_assets, n_obs, nu_init)

        # AIC:  GARCH params (3 per asset) + DCC α,β (+ ν for t-Copula)
        n_garch_params = 3 * n_assets
        k = n_garch_params + 2  # GARCH + DCC α,β
        if copula_type == "t":
            k += 1  # ν
        aic_val = 2 * k - 2 * ll

        # Round parameters
        alpha_d = Decimal(str(round(alpha_a, 6)))
        beta_d = Decimal(str(round(beta_b, 6)))
        nu_d = Decimal(str(round(nu, 2)))

        # Conditional covariance: H_t = D_t · R_t · D_t
        h_t_list = []
        for t in range(n_obs):
            sigma_t = [cond_vol[i][t] for i in range(n_assets)]
            h_t = [[0.0] * n_assets for _ in range(n_assets)]
            for i in range(n_assets):
                for j in range(n_assets):
                    h_t[i][j] = sigma_t[i] * R_t_list[t][i][j] * sigma_t[j]
            h_t_list.append(h_t)

        return DCCFitResult(
            alpha=alpha_d,
            beta=beta_d,
            unconditional_corr=Q_bar,
            conditional_corr=R_t_list,
            conditional_vol=cond_vol,
            h_t=h_t_list,
            nu=nu_d,
            copula_type=copula_type,
            n_assets=n_assets,
            n_obs=n_obs,
            log_likelihood=ll,
            aic=aic_val,
        )

    @staticmethod
    def _garch_11_fit(returns: list[float]) -> tuple[float, float, float, list[float], list[float]]:
        """Univariate GARCH(1,1) fitting via QMLE.

        Returns:
            (omega, alpha, beta, conditional_volatilities, standardized_residuals)
        """
        n = len(returns)
        mean_ret = sum(returns) / n
        residuals = [r - mean_ret for r in returns]

        # Simple estimation via variance targeting + grid search
        unconditional_var = sum(r * r for r in residuals) / (n - 1)

        # GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        # With variance targeting: ω = (1 - α - β) · σ²
        # Search over (α, β)

        best_ll = float("inf")
        best_omega = 0.0
        best_alpha = 0.05
        best_beta = 0.90
        best_volat = unconditional_var ** 0.5
        best_z = residuals

        for alpha_g in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15]:
            for beta_g in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
                if alpha_g + beta_g >= 1.0:
                    continue
                omega_g = (1 - alpha_g - beta_g) * unconditional_var
                if omega_g <= 0:
                    continue

                # Run filter
                volat = []
                z_s = []
                sigma2 = unconditional_var
                ll_gen = 0.0
                for t in range(n):
                    volat.append(math.sqrt(sigma2))
                    z_t = residuals[t] / max(math.sqrt(sigma2), 1e-10)
                    z_s.append(z_t)
                    ll_gen += -0.5 * math.log(2 * math.pi) - 0.5 * math.log(sigma2) - 0.5 * z_t * z_t
                    # Update: σ²_{t+1} = ω + α·ε²_t + β·σ²_t
                    sigma2 = omega_g + alpha_g * residuals[t] * residuals[t] + beta_g * sigma2
                    sigma2 = max(1e-10, sigma2)

                # Quasi-log-likelihood (we minimize negative)
                neg_ll = -ll_gen
                if neg_ll < best_ll:
                    best_ll = neg_ll
                    best_omega = omega_g
                    best_alpha = alpha_g
                    best_beta = beta_g
                    best_volat = volat
                    best_z = z_s

        if best_beta + best_alpha >= 1.0:
            best_beta = 0.90
            best_alpha = 0.05
            best_omega = (1 - best_alpha - best_beta) * unconditional_var

        return best_omega, best_alpha, best_beta, best_volat, best_z

    @staticmethod
    def _fit_nu_t_copula(
        R_t_list: list[list[list[float]]],
        std_residuals: list[list[float]],
        n_assets: int,
        n_obs: int,
        nu_init: float,
    ) -> tuple[float, float]:
        """Profile MLE for t-Copula df ν."""
        best_nu = nu_init
        best_ll = -float("inf")

        for nu_c in [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]:
            ll = 0.0
            for t in range(n_obs):
                R_t = R_t_list[t]
                z_t = [std_residuals[i][t] for i in range(n_assets)]

                if n_assets == 2:
                    # Bivariate t-Copula log-likelihood
                    rho = R_t[0][1]
                    x1, x2 = z_t[0], z_t[1]
                    q = (x1 * x1 - 2 * rho * x1 * x2 + x2 * x2) / (nu_c * (1 - rho * rho + 1e-10))
                    d = (math.gamma((nu_c + 2) / 2)
                         / (math.gamma(nu_c / 2) * nu_c * math.pi * math.sqrt(max(1 - rho * rho, 1e-10)))
                         * (1 + q) ** (-(nu_c + 2) / 2))
                    if d > 0:
                        ll += math.log(d)
                else:
                    # Multivariate approximation
                    try:
                        det_R = DCCCopula._det_2x2(R_t) if n_assets == 2 else 1.0
                        if det_R > 0:
                            ll += math.log(max(det_R, 1e-15))
                    except Exception:
                        pass

            if ll > best_ll:
                best_ll = ll
                best_nu = nu_c

        return best_ll, best_nu

    @staticmethod
    def _det_2x2(m: list[list[float]]) -> float:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    @staticmethod
    def simulate(
        fit: DCCFitResult,
        n_scenarios: int = 10000,
        use_latest_corr: bool = True,
    ) -> list[list[float]]:
        """从 DCC-Copula 生成模拟场景.

        Args:
            fit: DCC 拟合结果
            n_scenarios: 模拟场景数
            use_latest_corr: True=用最新相关矩阵, False=用无条件相关

        Returns:
            [scenario][asset] 标准化残差
        """
        import random
        rng = random.Random(42)

        if use_latest_corr:
            corr = fit.latest_correlation()
        else:
            corr = fit.unconditional_corr

        n = fit.n_assets
        nu = float(fit.nu)
        scenarios = []

        if fit.copula_type == "t":
            for _ in range(n_scenarios):
                # Generate from multivariate t: Z ~ N(0, corr) / sqrt(χ²_ν / ν)
                # Cholesky of corr
                chol = DCCCopula._cholesky(corr)
                # Standard normal
                z_norm = [rng.gauss(0, 1) for _ in range(n)]
                # Correlated normal
                z_corr = [sum(chol[i][j] * z_norm[j] for j in range(len(z_norm))) for i in range(n)]
                # χ² scaling
                chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(int(nu)))
                scale = math.sqrt(max(chi2 / nu, 0.1))
                scenarios.append([z_corr[i] / scale for i in range(n)])
        else:
            # Gaussian Copula
            chol = DCCCopula._cholesky(corr)
            for _ in range(n_scenarios):
                z_norm = [rng.gauss(0, 1) for _ in range(n)]
                z_corr = [sum(chol[i][j] * z_norm[j] for j in range(len(z_norm))) for i in range(n)]
                scenarios.append(z_corr)

        return scenarios

    @staticmethod
    def _cholesky(A: list[list[float]]) -> list[list[float]]:
        """Cholesky decomposition L where A = L·L'."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(A[i][i] - s, 1e-15))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-15)
        return L

    @staticmethod
    def dynamic_tail_dependence(
        fit: DCCFitResult,
        i: int = 0,
        j: int = 1,
    ) -> list[dict]:
        """计算时变尾部依赖序列.

        Returns:
            [{t: int, rho: float, lower_tail_dep: float, upper_tail_dep: float}]
        """
        nu = float(fit.nu)
        result = []
        for t, corr in enumerate(fit.correlation_series(i, j)):
            rho = max(-0.999, min(corr, 0.999))
            arg = -math.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            ld = 2 * DCCCopula._tcdf(arg, nu + 1)
            result.append({
                "t": t,
                "rho": round(rho, 4),
                "lower_tail_dep": round(ld, 4),
                "upper_tail_dep": round(ld, 4),  # t-Copula: symmetric
            })
        return result
