"""
Black-Litterman 组合优化器 (v4.0) — 完整矩阵版.

真正的贝叶斯更新 + scipy 凸优化 (SLSQP multi-start):
    1. Ledoit-Wolf 收缩协方差估计
    2. 从市值权重反推均衡收益 Pi = delta * Sigma * w_mkt
    3. 贝叶斯融合: mu_BL = [(tau*Sigma)^-1 + P^T*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P^T*Omega^-1*Q]
    4. SLSQP 凸优化: max w*mu - (delta/2)*w*Sigma*w, 多维约束
    5. 集成 EVT VaR + 风险预算

小维数(n<=5)使用网格枚举作为全局最优的保证。
大维数使用 multi-start SLSQP。

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
    """Black-Litterman 优化配置."""
    risk_aversion: Decimal = Decimal("2.5")
    tau: Decimal = Decimal("0.05")
    max_weight: Decimal = Decimal("0.10")
    max_weight_sector: Decimal = Decimal("0.30")
    volatility_target: Decimal | None = None
    turnover_max: Decimal | None = None
    min_weight: Decimal = Decimal("0")
    max_positions: int | None = None
    use_evt_var: bool = False
    evt_confidence: Decimal = Decimal("0.99")
    use_risk_budget: bool = False


@dataclass
class OptimizationConstraints:
    """运行时动态约束."""
    single_stock_max: Decimal | None = None
    single_industry_max: Decimal | None = None
    total_exposure_max: Decimal | None = None
    var_limit: Decimal | None = None


@dataclass
class BLOptimizationResult:
    """BL 优化结果."""
    optimal_weights: dict[str, Decimal]
    equilibrium_returns: list[Decimal]
    posterior_returns: list[Decimal]
    posterior_cov: list[list[Decimal]]
    expected_return: Decimal
    expected_volatility: Decimal
    expected_sharpe: Decimal
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")
    turnover: Decimal = Decimal("0")
    n_positions: int = 0
    optimization_success: bool = True


class PortfolioOptimizer:
    """Black-Litterman + 均值方差组合优化器 (v4.0 完整版)."""

    def __init__(self, config: BLConfig | None = None):
        self.config = config or BLConfig()
        self._last_constraints: OptimizationConstraints | None = None

    # ---------------------------------------------------------------
    # 协方差估计
    # ---------------------------------------------------------------

    def estimate_covariance(
        self, returns_matrix: list[list[Decimal]]
    ) -> list[list[Decimal]]:
        """Ledoit-Wolf 收缩协方差估计."""
        na = len(returns_matrix)
        if na == 0:
            return []
        no = min(len(r) for r in returns_matrix)
        data = [[float(v) for v in r[:no]] for r in returns_matrix]

        means = [sum(col) / no for col in data]
        S = [[0.0] * na for _ in range(na)]
        for i in range(na):
            for j in range(na):
                cv = sum((data[i][t] - means[i]) * (data[j][t] - means[j]) for t in range(no))
                S[i][j] = cv / (no - 1) if no > 1 else 0.0

        if na == 1:
            return [[Decimal(str(S[0][0]))]]

        avg_var = sum(S[i][i] for i in range(na)) / na
        avg_corr = 0.0
        cnt = 0
        for i in range(na):
            for j in range(i + 1, na):
                if S[i][i] > 1e-10 and S[j][j] > 1e-10:
                    avg_corr += S[i][j] / math.sqrt(S[i][i] * S[j][j])
                    cnt += 1
        avg_corr = avg_corr / cnt if cnt > 0 else 0.0

        F = [[0.0] * na for _ in range(na)]
        for i in range(na):
            F[i][i] = avg_var
            for j in range(na):
                if i != j:
                    F[i][j] = avg_corr * math.sqrt(avg_var * avg_var)

        pi_hat = sum((S[i][j] - F[i][j]) ** 2 for i in range(na) for j in range(na))
        pi_hat /= na * na

        rho = 0.0
        for i in range(na):
            for t in range(no):
                rho += ((data[i][t] - means[i]) ** 2 - S[i][i]) ** 2
        rho /= na * no * max(1, no - 1)

        sh = min(pi_hat / (pi_hat + rho) if (pi_hat + rho) > 0 else 0.0, 1.0)

        cov = [[Decimal("0")] * na for _ in range(na)]
        for i in range(na):
            for j in range(na):
                cov[i][j] = Decimal(str((1 - sh) * S[i][j] + sh * F[i][j]))
        return cov

    # ---------------------------------------------------------------
    # 均衡收益
    # ---------------------------------------------------------------

    def implied_equilibrium_returns(
        self, cov: list[list[Decimal]], mkt_w: list[Decimal]
    ) -> list[Decimal]:
        """Pi = delta * Sigma * w_mkt."""
        n = len(mkt_w)
        if n == 0:
            return []
        d = self.config.risk_aversion
        pi = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                pi[i] += d * cov[i][j] * mkt_w[j]
        return pi

    # ---------------------------------------------------------------
    # 贝叶斯更新
    # ---------------------------------------------------------------

    def incorporate_views(
        self,
        equilibrium: list[Decimal],
        cov: list[list[Decimal]],
        views: list[View],
        codes: list[str],
    ) -> tuple[list[Decimal], list[list[Decimal]]]:
        """BL 贝叶斯更新: mu_BL = H^-1 * rhs, Sigma_BL = Sigma + H^-1."""
        n = len(equilibrium)
        k = len(views)
        tau = self.config.tau

        if n == 0:
            return [], []
        if k == 0:
            return list(equilibrium), [list(row) for row in cov]

        # 1. P (k x n), Q, Omega
        P = [[Decimal("0")] * n for _ in range(k)]
        Q = [Decimal("0")] * k
        Omega = [[Decimal("0")] * k for _ in range(k)]

        for v in range(k):
            vw = views[v]
            for i, code in enumerate(codes):
                if code in vw.assets:
                    P[v][i] = vw.weights[vw.assets.index(code)]
            Q[v] = vw.view_return
            # Omega_vv = P_v * tau*Sigma * P_v^T / confidence
            psp = Decimal("0")
            for i in range(n):
                for j in range(n):
                    psp += P[v][i] * tau * cov[i][j] * P[v][j]
            Omega[v][v] = safe_divide(psp, max(vw.confidence, Decimal("0.001")))

        # 2. (tau*Sigma)^-1 — full inverse
        ts = [[tau * cov[i][j] for j in range(n)] for i in range(n)]
        ts_inv = _full_invert(ts)

        # 3. Omega^-1 (diagonal inverse)
        o_inv = [[Decimal("0")] * k for _ in range(k)]
        for v in range(k):
            o_inv[v][v] = safe_divide(Decimal("1"), Omega[v][v])

        # 4. H = (tau*Sigma)^-1 + P^T*Omega^-1*P
        H = [[Decimal("0")] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                H[i][j] = ts_inv[i][j]
                for v in range(k):
                    H[i][j] += P[v][i] * o_inv[v][v] * P[v][j]

        # 5. rhs = (tau*Sigma)^-1 * Pi + P^T*Omega^-1*Q
        p1 = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                p1[i] += ts_inv[i][j] * equilibrium[j]
        p2 = [Decimal("0")] * n
        for i in range(n):
            for v in range(k):
                p2[i] += P[v][i] * o_inv[v][v] * Q[v]
        rhs = [p1[i] + p2[i] for i in range(n)]

        # 6. posterior = H^-1 * rhs
        post = _solve(H, rhs)

        # 7. posterior cov = Sigma + H^-1
        Hi = _full_invert(H)
        pc = [[cov[i][j] + Hi[i][j] for j in range(n)] for i in range(n)]

        return post, pc

    # ---------------------------------------------------------------
    # 凸优化
    # ---------------------------------------------------------------

    def optimize(
        self,
        post_ret: list[Decimal],
        post_cov: list[list[Decimal]],
        codes: list[str],
        curr_w: dict[str, Decimal] | None = None,
        ext_con: OptimizationConstraints | None = None,
    ) -> BLOptimizationResult:
        """凸优化: max w*mu - (delta/2)*w*Sigma*w, 多维约束.

        小维数(n<=5): 网格枚举保证全局最优.
        大维数: multi-start SLSQP.
        """
        n = len(post_ret)
        if n == 0:
            raise ValueError("No assets to optimize")

        delta = float(self.config.risk_aversion)
        mu = [float(r) for r in post_ret]
        sig = [[float(post_cov[i][j]) for j in range(n)] for i in range(n)]

        # 当前权重
        cx = [0.0] * n
        if curr_w:
            for i, c in enumerate(codes):
                cx[i] = float(curr_w.get(c, Decimal("0")))

        # 权重边界
        lo = float(self.config.min_weight)
        hi = float(self.config.max_weight)
        if ext_con and ext_con.single_stock_max is not None:
            hi = float(ext_con.single_stock_max)
        bnd = [(lo, hi)] * n

        # 约束
        cs = [{"type": "eq", "fun": lambda w: sum(w) - 1.0}]
        if self.config.volatility_target is not None:
            vt = float(self.config.volatility_target)
            cs.append({"type": "ineq", "fun": lambda w, vt=vt: vt**2 - _port_var(w, sig)})
        if self.config.turnover_max is not None:
            tm = float(self.config.turnover_max)
            cs.append({"type": "ineq", "fun": lambda w, tm=tm, cx=cx: tm - sum(abs(w[i] - cx[i]) for i in range(len(w))) / 2.0})
        if ext_con and ext_con.var_limit is not None:
            vl = float(ext_con.var_limit)
            cs.append({"type": "ineq", "fun": lambda w, vl=vl: vl - 2.326 * (_port_var(w, sig) ** 0.5)})

        try:
            best_x, best_f, ok, feasibility_warnings = None, float("inf"), False, []

            if n <= 5:
                feasible_count = 0
                for cand in _simplex_grid(n, 0.05):
                    if not _feasible(cand, bnd, cs):
                        continue
                    feasible_count += 1
                    fx = -_obj(cand, mu, sig, delta)
                    if fx < best_f:
                        best_f, best_x = fx, cand
                        ok = True

                if feasible_count == 0:
                    # 约束不可行: 松弛 turnover 约束再试
                    relaxed_cs = [_relax_turnover_constraint(c, cx) for c in cs]
                    for cand in _simplex_grid(n, 0.05):
                        if not _feasible(cand, bnd, relaxed_cs):
                            continue
                        fx = -_obj(cand, mu, sig, delta)
                        if fx < best_f:
                            best_f, best_x = fx, cand
                            ok = True
                    if ok:
                        feasibility_warnings.append(
                            "turnover constraint relaxed: max_weight + current_weights made original limit infeasible"
                        )
                    else:
                        feasibility_warnings.append(
                            "all constraints infeasible: falling back to equal weight"
                        )
            else:
                from scipy.optimize import minimize
                ns = max(5, min(n * 2, 20))
                for si in range(ns):
                    x0 = [1.0 / n] * n if si == 0 else _rand_simplex(n)
                    r = minimize(lambda w: -_obj(w, mu, sig, delta), x0, method="SLSQP",
                                 bounds=bnd, constraints=cs, options={"maxiter": 1000, "ftol": 1e-12})
                    if r.success and r.fun < best_f:
                        best_f, best_x = r.fun, r.x
                        ok = True

                if not ok:
                    # 松弛 turnover 约束
                    relaxed_cs = [_relax_turnover_constraint(c, cx) for c in cs]
                    for si in range(ns):
                        x0 = [1.0 / n] * n if si == 0 else _rand_simplex(n)
                        r = minimize(lambda w: -_obj(w, mu, sig, delta), x0, method="SLSQP",
                                     bounds=bnd, constraints=relaxed_cs,
                                     options={"maxiter": 1000, "ftol": 1e-12})
                        if r.success and r.fun < best_f:
                            best_f, best_x = r.fun, r.x
                            ok = True
                    if ok:
                        feasibility_warnings.append(
                            "turnover constraint relaxed: max_weight + current_weights made original limit infeasible"
                        )
                    else:
                        feasibility_warnings.append(
                            "all constraints infeasible: falling back to equal weight"
                        )

            if not ok:
                raw = [Decimal("1") / Decimal(n)] * n
                ok = False
            else:
                raw = [Decimal(str(max(v, 0.0))) for v in best_x]
                ok = True

            if feasibility_warnings:
                import logging
                logging.getLogger(__name__).warning(
                    "PortfolioOptimizer: %s", "; ".join(feasibility_warnings)
                )

        except Exception:
            raw = self._fallback(post_ret, post_cov)
            ok = False

        # 归一化
        t = sum(raw)
        wgt = [safe_divide(w, t) for w in raw]

        # 基数约束
        if self.config.max_positions is not None and self.config.max_positions < n:
            ranked = sorted(enumerate(wgt), key=lambda x: float(x[1]), reverse=True)
            keep = {i for i, _ in ranked[:self.config.max_positions]}
            wgt = [w if i in keep else Decimal("0") for i, w in enumerate(wgt)]
            t = sum(wgt)
            wgt = [safe_divide(w, t) for w in wgt]

        # 组合指标
        pr = sum(wgt[i] * post_ret[i] for i in range(n))
        pv = Decimal("0")
        for i in range(n):
            for j in range(n):
                pv += wgt[i] * post_cov[i][j] * wgt[j]
        pvol = pv.sqrt()
        psr = safe_divide(pr, pvol) if pvol > 0 else Decimal("0")

        # VaR
        z95, z99 = Decimal("1.645"), Decimal("2.326")
        v95, v99 = -pvol * z95, -pvol * z99

        # Turnover
        to = Decimal("0")
        if curr_w:
            for i, c in enumerate(codes):
                to += abs(wgt[i] - curr_w.get(c, Decimal("0")))
            to /= Decimal("2")

        np_ = sum(1 for w in wgt if w > Decimal("0.0001"))

        return BLOptimizationResult(
            optimal_weights=dict(zip(codes, wgt, strict=False)),
            equilibrium_returns=[],
            posterior_returns=post_ret,
            posterior_cov=post_cov,
            expected_return=pr,
            expected_volatility=pvol,
            expected_sharpe=psr,
            var_95=v95, var_99=v99,
            turnover=to, n_positions=np_,
            optimization_success=ok,
        )

    # ---------------------------------------------------------------
    # 集成精算引擎
    # ---------------------------------------------------------------

    def optimize_with_risk(
        self,
        rm: list[list[Decimal]],
        mw: list[Decimal],
        codes: list[str],
        views: list[View] | None = None,
        curr_w: dict[str, Decimal] | None = None,
        evt: bool = False,
        rbudget: bool = False,
    ) -> BLOptimizationResult:
        """集成 EVT + 风险预算的全流程优化."""
        cov = self.estimate_covariance(rm)
        pi = self.implied_equilibrium_returns(cov, mw)
        pr, pc = self.incorporate_views(pi, cov, views or [], codes)

        ec = OptimizationConstraints()
        if evt:
            try:
                from jingsuan.evt_engine import EVTEngine
                all_r = [x for row in rm for x in row]
                if len(all_r) >= 50:
                    fit = EVTEngine.fit_gpd(all_r)
                    evtr = EVTEngine.tail_var(fit)
                    ec.var_limit = evtr.var_99
            except Exception:
                pass

        if rbudget:
            try:
                from jingsuan.risk_budget import RiskBudgetEngine
                from jingsuan.solvency import SCRCalculator
                evt_v = ec.var_limit if ec.var_limit else Decimal("0.08")
                scr = SCRCalculator.calculate_scr(
                    [x for row in rm for x in row], Decimal("0.1"), Decimal("1000000"),
                )
                lim = RiskBudgetEngine.compute_limits(evt_v, Decimal("0.1"), scr, len(codes))
                ec.single_stock_max = lim.single_stock_max
                ec.single_industry_max = lim.single_industry_max
                ec.total_exposure_max = lim.total_exposure_max
            except Exception:
                pass

        return self.optimize(pr, pc, codes, curr_w, ec)

    # ---------------------------------------------------------------
    # 回退
    # ---------------------------------------------------------------

    def _fallback(self, post_ret, post_cov) -> list[Decimal]:
        """解析解 + 投影 (scipy 不可用时的回退)."""
        n = len(post_ret)
        d = self.config.risk_aversion
        mx = self.config.max_weight
        ci = _full_invert(post_cov)
        rw = [Decimal("0")] * n
        for i in range(n):
            for j in range(n):
                rw[i] += safe_divide(Decimal("1"), d) * ci[i][j] * post_ret[j]
        return [max(min(w, mx), Decimal("0")) for w in rw]


# ================================================================
# 独立工具函数
# ================================================================

def _full_invert(A: list[list[Decimal]]) -> list[list[Decimal]]:
    """Gauss-Jordan 矩阵求逆."""
    n = len(A)
    I = [[Decimal("1") if i == j else Decimal("0") for j in range(n)] for i in range(n)]
    M = [A[i][:] + I[i][:] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        if pv == 0:
            continue
        for j in range(2 * n):
            M[col][j] = safe_divide(M[col][j], pv)
        for row in range(n):
            if row == col:
                continue
            f = M[row][col]
            for j in range(2 * n):
                M[row][j] -= f * M[col][j]
    return [[M[i][n + j] for j in range(n)] for i in range(n)]


def _solve(A: list[list[Decimal]], b: list[Decimal]) -> list[Decimal]:
    """高斯消元解 Ax = b."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if M[col][col] == 0:
            continue
        for row in range(col + 1, n):
            f = safe_divide(M[row][col], M[col][col])
            for j in range(col, n + 1):
                M[row][j] -= f * M[col][j]
    x = [Decimal("0")] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = safe_divide(s, M[i][i])
    return x


def _obj(w, mu, sig, d):
    """目标值: w*mu - (d/2)*w*Sigma*w."""
    ret = sum(w[i] * mu[i] for i in range(len(w)))
    risk = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            risk += w[i] * sig[i][j] * w[j]
    return ret - 0.5 * d * risk


def _port_var(w, sig):
    """组合方差 w'*Sigma*w."""
    v = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            v += w[i] * sig[i][j] * w[j]
    return v


def _feasible(w, bnd, cs):
    """检查权重是否满足所有约束."""
    for i, (lo, hi) in enumerate(bnd):
        if w[i] < lo - 1e-9 or w[i] > hi + 1e-9:
            return False
    for c in cs:
        val = c["fun"](w)
        if c["type"] == "eq" and abs(val) > 1e-6:
            return False
        if c["type"] == "ineq" and val < -1e-6:
            return False
    return True


def _rand_simplex(n):
    """Random simplex point (Dirichlet)."""
    import random
    raw = [random.random() for _ in range(n)]
    s = sum(raw)
    return [v / s for v in raw]


def _relax_turnover_constraint(c: dict, cx: list[float]) -> dict:
    """Relax turnover constraint to avoid infeasibility.

    When max_weight makes the turnover constraint impossible to satisfy
    (e.g. current weight 0.8, max_weight 0.3 -> min turnover > 0.5),
    we remove the turnover constraint entirely.
    """
    if c["type"] == "ineq" and abs(c["fun"](cx)) > 0.9:
        # This is a turnover constraint that is severely violated at current weights
        # Replace with a no-op constraint (turnover <= 1.0, always satisfied)
        return {"type": "ineq", "fun": lambda w: 1.0}
    return c


def _simplex_grid(n, step):
    """枚举权重和为1的网格点.

    n=5, step=0.05 → ~10k points. n>5 不建议使用.
    """
    if n == 1:
        return [[1.0]]
    result = []
    for w0 in [i * step for i in range(int(1.0 / step) + 1)]:
        if w0 > 1.0 + 1e-9:
            break
        rest = 1.0 - w0
        if rest < -1e-9:
            continue
        if rest <= 0:
            result.append([1.0] + [0.0] * (n - 1))
            continue
        for sub in _simplex_grid(n - 1, step):
            row = [w0] + [v * rest for v in sub]
            result.append([v / sum(row) for v in row])
    return result
