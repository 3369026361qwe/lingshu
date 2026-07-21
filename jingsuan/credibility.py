"""
信度理论引擎 — Bühlmann-Straub + Hierarchical + Hachemeister (v4.1).

纯计算层：无状态、无 IO。
将三路信号 (因子/GNN/Agent) 的线性 IC 加权升级为信度加权，
自动考虑每个信号源的时序稳定性和覆盖广度。

v4.1 增强:
    - 分层信度模型 (Hierarchical Credibility)
    - 回归信度 (Hachemeister 模型)
    - 信度因子的时序衰减加权

数学基础:
    Bühlmann-Straub:
        μ̂_i = Z_i · X̄_i^w + (1 - Z_i) · μ
        Z_i = Σw / (Σw + m/v)

    Hachemeister (Regression Credibility):
        Y_i = X_i · b_i + ε_i
        b_i ~ N(b, A)  (random effects across sources)
        b̂_i = Z_i · b̂_i^(ind) + (I - Z_i) · b̂^(pooled)
        其中 Z_i = A · (A + s²·(X_i'X_i)^{-1})^{-1}

    Time-Decay Weighting:
        w_t = λ^{T-t}  (exponential decay)
        Z_i(t) = Σ_{s=1}^{t} w_s / (Σ w_s + m/v)

Usage:
    from jingsuan import CredibilityEngine, SourceTrackRecord
    weights = CredibilityEngine.buhlmann_straub(sources)
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide, safe_mean

# ── Dataclasses ───────────────────────────────────────────

@dataclass
class SourceTrackRecord:
    """单个信号源的历史表现."""
    name: str
    ic_values: list[Decimal]
    weights: list[Decimal] | None = None  # 覆盖广度/成交额
    dates: list[str] | None = None

    @property
    def n_periods(self) -> int:
        return len(self.ic_values)

    @property
    def mean_ic(self) -> Decimal:
        return safe_mean(self.ic_values)

    @property
    def var_ic(self) -> Decimal:
        n = len(self.ic_values)
        if n < 2:
            return Decimal("0")
        mu = safe_mean(self.ic_values)
        return sum((x - mu) ** 2 for x in self.ic_values) / Decimal(n - 1)


@dataclass
class CredibilityWeights:
    source_weights: dict[str, Decimal]       # 融合权重 (总和=1)
    credibility_factors: dict[str, Decimal]  # 信度因子 Z_i
    posterior_ic: dict[str, Decimal]         # 后验 IC 估计
    process_variance: Decimal                # m (within-source)
    hypothesis_variance: Decimal             # v (between-source)


@dataclass
class HierarchicalCredibilityResult:
    """分层信度模型结果."""
    level: str                               # "overall" | "regime_{k}"
    weights: CredibilityWeights              # 该层的融合权重
    regime_labels: list[str] | None = None   # 体制标签
    regime_weights: dict[str, CredibilityWeights] | None = None  # 体制内权重
    overall_blend: dict[str, Decimal] | None = None  # 跨体制混合权重


@dataclass
class HachemeisterResult:
    """回归信度 (Hachemeister) 结果."""
    source_predictions: dict[str, Decimal]   # 各源回归预测
    pooled_prediction: Decimal               # 混合模型预测
    blended_predictions: dict[str, Decimal]  # 信度加权混合预测
    credibility_matrices: dict[str, list[list[Decimal]]]  # 信度矩阵 Z_i
    regression_params: dict[str, dict[str, Decimal]]  # {source: {param: value}}


@dataclass
class TimeDecayConfig:
    """时序衰减配置."""
    decay_factor: Decimal = Decimal("0.95")  # λ (per period)
    half_life_periods: int = 14              # t_{1/2}


# ── Credibility Engine v4.1 ───────────────────────────────

class CredibilityEngine:
    """信度理论引擎 — Bühlmann-Straub / Hierarchical / Hachemeister 多源融合。"""

    # ═══════════════════════════════════════════════════════
    # Bühlmann-Straub (Core)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def buhlmann_straub(
        sources: list[SourceTrackRecord],
        decay_config: TimeDecayConfig | None = None,
    ) -> CredibilityWeights:
        """Bühlmann-Straub 信度模型 (v4.1: + 时序衰减).

        Args:
            sources: 信号源列表
            decay_config: 时序衰减配置 (None = 无衰减)
        """
        if len(sources) < 2:
            raise ValueError(f"需要 >= 2 个信号源, 当前 {len(sources)}")

        # 1. 总体均值 μ
        all_ics = []
        for s in sources:
            all_ics.extend(s.ic_values)
        mu = safe_mean(all_ics) if all_ics else Decimal("0")

        # 2. 过程方差 m (within-source)
        within_vars = []
        for s in sources:
            if s.n_periods < 2:
                continue
            mean_i = s.mean_ic
            var_i = sum((x - mean_i) ** 2 for x in s.ic_values) / Decimal(s.n_periods - 1)
            within_vars.append(var_i)
        m = safe_mean(within_vars) if within_vars else Decimal("1")

        # 3. 假设均值方差 v (between-source)
        source_means = [s.mean_ic for s in sources]
        mean_of_means = safe_mean(source_means)
        n_s = Decimal(len(sources))
        if n_s > 1:
            v = sum((sm - mean_of_means) ** 2 for sm in source_means) / (n_s - 1)
        else:
            v = Decimal("0")
        if v <= 0:
            v = m * Decimal("0.1")

        # 4. 信度因子 Z_i (含时序衰减)
        z_factors = {}
        weighted_means = {}
        effective_n = {}

        for s in sources:
            n = s.n_periods

            if decay_config and n > 1:
                # Exponential decay weighting
                lam = float(decay_config.decay_factor)
                weights_decayed = [lam ** (n - 1 - t) for t in range(n)]

                total_w = sum(weights_decayed)
                if total_w > 0:
                    weighted_mean = sum(
                        float(s.ic_values[t]) * weights_decayed[t] for t in range(n)
                    ) / total_w
                else:
                    weighted_mean = float(s.mean_ic)

                # Effective sample size with decay
                n_eff = (1 - lam ** n) / (1 - lam) if lam != 1 else n
                effective_n[s.name] = Decimal(str(n_eff))
                weighted_means[s.name] = Decimal(str(weighted_mean))
            else:
                if s.weights and len(s.weights) == n:
                    total_w = sum(s.weights)
                    if total_w > 0:
                        weighted_mean = sum(
                            ic * w for ic, w in zip(s.ic_values, s.weights, strict=False)
                        ) / total_w
                    else:
                        weighted_mean = s.mean_ic
                else:
                    total_w = Decimal(n)
                    weighted_mean = s.mean_ic
                effective_n[s.name] = Decimal(n)
                weighted_means[s.name] = weighted_mean

            z_i = safe_divide(effective_n[s.name], effective_n[s.name] + safe_divide(m, v))
            z_factors[s.name] = z_i

        # 5. 后验估计
        posterior_ic = {}
        for s in sources:
            posterior_ic[s.name] = (
                z_factors[s.name] * weighted_means[s.name]
                + (Decimal("1") - z_factors[s.name]) * mu
            )

        # 6. 融合权重: softmax(posterior_IC / sigma)
        posterior_vars = {}
        for s in sources:
            posterior_vars[s.name] = safe_divide(m, effective_n[s.name] + safe_divide(m, v))

        scores = {}
        for s in sources:
            sigma_hat = posterior_vars[s.name].sqrt()
            if sigma_hat > 0:
                ratio = float(posterior_ic[s.name] / sigma_hat)
                ratio = max(-50.0, min(ratio, 50.0))  # prevent overflow
                scores[s.name] = Decimal(str(math.exp(ratio)))
            else:
                scores[s.name] = Decimal("1")

        total_score = sum(scores.values())
        if total_score > 0:
            weights = {
                name: safe_divide(score, total_score)
                for name, score in scores.items()
            }
        else:
            n_sources = len(sources)
            w = Decimal("1") / Decimal(n_sources)
            weights = {s.name: w for s in sources}

        return CredibilityWeights(
            source_weights=weights,
            credibility_factors=z_factors,
            posterior_ic=posterior_ic,
            process_variance=m,
            hypothesis_variance=v,
        )

    @staticmethod
    def fuse_signals(sources: list[SourceTrackRecord]) -> list[Decimal]:
        """直接返回融合权重列表."""
        result = CredibilityEngine.buhlmann_straub(sources)
        return [
            result.source_weights.get(s.name, Decimal("0"))
            for s in sources
        ]

    # ═══════════════════════════════════════════════════════
    # Hierarchical Credibility
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def hierarchical_credibility(
        sources: list[SourceTrackRecord],
        regime_labels: list[int] | None = None,
        regime_names: dict[int, str] | None = None,
    ) -> HierarchicalCredibilityResult:
        """分层信度模型.

        两层结构:
            Level 1 (Within-regime): 在每个市场体制内各自计算 Bühlmann-Straub
            Level 2 (Between-regime): 体制间混合

        Args:
            sources: 信号源列表
            regime_labels: 每个时期的体制标签 (长度 = 每个源的最长观测)
            regime_names: {label_id: "bull"/"bear"/"neutral"}

        Returns:
            HierarchicalCredibilityResult
        """
        if regime_labels is None:
            # No regimes → fall through to basic Bühlmann-Straub
            base_weights = CredibilityEngine.buhlmann_straub(sources)
            return HierarchicalCredibilityResult(
                level="overall",
                weights=base_weights,
            )

        if regime_names is None:
            unique_labels = sorted(set(regime_labels))
            regime_names = {lbl: f"regime_{lbl}" for lbl in unique_labels}

        unique_regimes = sorted(set(regime_labels))
        regime_sources = {}

        # Split each source's track record by regime
        for lbl in unique_regimes:
            regime_sources[lbl] = []
            for s in sources:
                # Extract observations belonging to this regime
                regime_ics = []
                regime_wts = []
                regime_dates = []
                for t, (ic, date) in enumerate(zip(s.ic_values, s.dates or [], strict=False)):
                    if t < len(regime_labels) and regime_labels[t] == lbl:
                        regime_ics.append(ic)
                        if s.weights and t < len(s.weights):
                            regime_wts.append(s.weights[t])
                        if s.dates:
                            regime_dates.append(date)

                if regime_ics:
                    regime_sources[lbl].append(SourceTrackRecord(
                        name=s.name,
                        ic_values=regime_ics,
                        weights=regime_wts if regime_wts else None,
                        dates=regime_dates if regime_dates else None,
                    ))

        # Compute within-regime credibility
        regime_weights = {}
        regime_overall_stats = {}
        for lbl in unique_regimes:
            if len(regime_sources[lbl]) >= 2:
                rw = CredibilityEngine.buhlmann_straub(regime_sources[lbl])
                regime_weights[regime_names[lbl]] = rw
                # Overall within-regime posterior IC
                regime_overall_stats[regime_names[lbl]] = {
                    "n_sources": len(regime_sources[lbl]),
                    "mean_ic": str(safe_mean(
                        [Decimal(str(pi)) for pi in rw.posterior_ic.values()]
                    )),
                }

        # Cross-regime blend: weight by how often each regime occurs
        regime_counts = {}
        for lbl in unique_regimes:
            regime_counts[lbl] = sum(1 for rl in regime_labels if rl == lbl)
        total_periods = len(regime_labels)

        overall_blend = {}
        for s in sources:
            blended = Decimal("0")
            total_weight = Decimal("0")
            for lbl in unique_regimes:
                rname = regime_names[lbl]
                if rname in regime_weights:
                    w = Decimal(str(regime_counts[lbl] / total_periods))
                    if s.name in regime_weights[rname].source_weights:
                        blended += w * regime_weights[rname].source_weights[s.name]
                        total_weight += w
            overall_blend[s.name] = safe_divide(blended, total_weight) if total_weight > 0 else Decimal("0")

        # Overall weights (across all regimes, using full data)
        base_weights = CredibilityEngine.buhlmann_straub(sources)

        return HierarchicalCredibilityResult(
            level="hierarchical",
            weights=base_weights,
            regime_labels=[regime_names[lbl] for lbl in unique_regimes],
            regime_weights=regime_weights,
            overall_blend=overall_blend,
        )

    # ═══════════════════════════════════════════════════════
    # Hachemeister Regression Credibility
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def hachemeister_regression(
        sources: list[SourceTrackRecord],
        features: list[list[Decimal]] | None = None,
    ) -> HachemeisterResult:
        """Hachemeister 回归信度模型.

        模型: Y_i = X_i · b_i + ε_i
               b_i ~ N(b, A)
               ε_i ~ N(0, s²·W_i^{-1})

        其中:
            Y_i = 信号源 i 的 IC 向量
            X_i = 设计矩阵 (特征: 滞后 IC, 波动率, 成交量等)
            b_i = 信号源 i 的回归系数
            b   = 混合回归系数 (先验)
            A   = 源间方差-协方差 (随机效应)
            s²  = 过程方差 (per-observation error variance)

        Simplification for 3 signal sources (typical in 灵枢):
            - X_i = [1, lagged_IC, vol_regime] (intercept + 1-2 features)
            - Pooled: OLS on all data
            - Individual: OLS per source
            - Blend: credibility-weighted combination

        Args:
            sources: 信号源
            features: 可选的外部特征 (若为 None, 使用滞后 IC + 波动率代理)

        Returns:
            HachemeisterResult
        """
        len(sources)

        # Build design matrix for each source
        # Y = IC_t, X = [1, IC_{t-1}] (simple AR(1) model)
        source_y = {}
        source_x = {}

        for s in sources:
            n = s.n_periods
            if n < 5:
                continue
            y = [float(ic) for ic in s.ic_values[1:]]
            # Design: [intercept, lagged_IC]
            x = [[1.0, float(s.ic_values[t])] for t in range(n - 1)]
            source_y[s.name] = y
            source_x[s.name] = x

        if len(source_y) < 2:
            # Fallback to basic Bühlmann-Straub
            base = CredibilityEngine.buhlmann_straub(sources)
            return HachemeisterResult(
                source_predictions={name: Decimal("0") for name in base.source_weights},
                pooled_prediction=Decimal("0"),
                blended_predictions={name: Decimal("0") for name in base.source_weights},
                credibility_matrices={},
                regression_params={},
            )

        # Individual OLS estimates for each source
        b_ind = {}  # b̂_i^(ind)
        s2_i = {}   # per-source residual variance
        for name in source_y:
            y_vec = source_y[name]
            x_mat = source_x[name]
            b, s2 = CredibilityEngine._ols(x_mat, y_vec)
            b_ind[name] = b
            s2_i[name] = s2

        # Pooled OLS (all sources combined)
        y_pooled = []
        x_pooled = []
        for name in source_y:
            y_pooled.extend(source_y[name])
            x_pooled.extend(source_x[name])
        b_pooled, s2_pooled = CredibilityEngine._ols(x_pooled, y_pooled)

        # Estimate between-source variance A
        # A = Cov(b_i) ≈ sample covariance of individual estimates
        k = len(b_pooled)  # number of regression coefficients
        b_ind_list = [b_ind[name] for name in b_ind]
        mean_b = [sum(b[i] for b in b_ind_list) / len(b_ind_list) for i in range(k)]
        A = [[0.0] * k for _ in range(k)]
        n_b = len(b_ind_list)
        if n_b >= 2:
            for r in range(k):
                for c in range(k):
                    A[r][c] = sum((b[r] - mean_b[r]) * (b[c] - mean_b[c]) for b in b_ind_list) / (n_b - 1)

        # Average per-observation variance s²
        s2_avg = sum(s2_i.values()) / max(1, len(s2_i))

        # Credibility matrices and blended predictions
        cred_matrices = {}
        blended_preds = {}

        for name in b_ind:
            # Z_i = A · (A + s²·(X_i'X_i)^{-1})^{-1}
            x_mat = source_x[name]
            n_obs = len(x_mat)

            # (X'X)^{-1}
            xtx = [[0.0] * k for _ in range(k)]
            for t in range(n_obs):
                for r in range(k):
                    for c in range(k):
                        xtx[r][c] += x_mat[t][r] * x_mat[t][c]
            xtx_inv = CredibilityEngine._inv_2x2(xtx) if k == 2 else [[1.0 / max(xtx[0][0], 1e-10)]]

            # A + s²·(X'X)^{-1}
            a_plus = [[0.0] * k for _ in range(k)]
            for r in range(k):
                for c in range(k):
                    a_plus[r][c] = A[r][c] + s2_avg * xtx_inv[r][c]

            # Z = A · (A + s²·(X'X)^{-1})^{-1}
            a_plus_inv = CredibilityEngine._inv_2x2(a_plus) if k == 2 else [[1.0 / max(a_plus[0][0], 1e-10)]]

            # Z_i matrix
            Z_i = [[0.0] * k for _ in range(k)]
            for r in range(k):
                for c in range(k):
                    Z_i[r][c] = sum(A[r][j] * a_plus_inv[j][c] for j in range(k))

            cred_matrices[name] = [[Decimal(str(round(z, 6))) for z in row] for row in Z_i]

            # Blended prediction: b̂_i = Z_i · b̂_i^(ind) + (I - Z_i) · b̂^(pooled)
            b_blend = []
            for r in range(k):
                br_ind = b_ind[name][r]
                br_pool = b_pooled[r]
                z_weight = Z_i[r][r]  # Use diagonal for simplicity
                b_blend.append(z_weight * br_ind + (1 - z_weight) * br_pool)
            blended_preds[name] = Decimal(str(round(b_blend[0], 6)))

        # Source predictions = expected IC (intercept term)
        source_preds = {name: Decimal(str(round(b_ind[name][0], 6))) for name in b_ind}
        pooled_pred = Decimal(str(round(b_pooled[0], 6)))

        # Regression params for each source
        reg_params = {}
        for name in b_ind:
            reg_params[name] = {
                "intercept": Decimal(str(round(b_ind[name][0], 6))),
                "ar_coef": Decimal(str(round(b_ind[name][1], 6))) if k >= 2 else Decimal("0"),
            }

        return HachemeisterResult(
            source_predictions=source_preds,
            pooled_prediction=pooled_pred,
            blended_predictions=blended_preds,
            credibility_matrices=cred_matrices,
            regression_params=reg_params,
        )

    # ═══════════════════════════════════════════════════════
    # Time-Decay Weighted Credibility
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def decay_weighted_credibility(
        sources: list[SourceTrackRecord],
        half_life: int = 14,
    ) -> CredibilityWeights:
        """时序衰减加权信度因子.

        使用指数衰减: w_t = λ^{T-t} where λ = 0.5^{1/half_life}.

        更近期的观测权重更大 → 信度因子更快反映信号源表现的近期变化。
        """
        lam = 0.5 ** (1.0 / half_life)
        config = TimeDecayConfig(
            decay_factor=Decimal(str(lam)),
            half_life_periods=half_life,
        )
        return CredibilityEngine.buhlmann_straub(sources, decay_config=config)

    # ═══════════════════════════════════════════════════════
    # Helper: OLS
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _ols(x: list[list[float]], y: list[float]) -> tuple[list[float], float]:
        """Ordinary Least Squares via normal equations.

        Returns (beta_coefficients, residual_variance).
        """
        n = len(y)
        k = len(x[0]) if x else 1
        if n < k:
            return [sum(y) / n] if n > 0 else [0.0], 1.0

        # X'X
        xtx = [[0.0] * k for _ in range(k)]
        for t in range(n):
            for i in range(k):
                for j in range(k):
                    xtx[i][j] += x[t][i] * x[t][j]

        # X'y
        xty = [0.0] * k
        for t in range(n):
            for i in range(k):
                xty[i] += x[t][i] * y[t]

        # Invert X'X (simple for k=2)
        if k == 2:
            det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0]
            if abs(det) < 1e-15:
                return [sum(y) / n, 0.0], 1.0
            inv = [
                [xtx[1][1] / det, -xtx[0][1] / det],
                [-xtx[1][0] / det, xtx[0][0] / det],
            ]
            beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
        else:
            # k=1: single intercept
            beta = [xty[0] / max(xtx[0][0], 1e-15)] if xtx[0][0] > 1e-15 else [sum(y) / n]

        # Residual variance
        residuals = [y[t] - sum(beta[j] * x[t][j] for j in range(k)) for t in range(n)]
        s2 = sum(r * r for r in residuals) / max(1, n - k)

        return beta, s2

    @staticmethod
    def _inv_2x2(a: list[list[float]]) -> list[list[float]]:
        """2x2 matrix inverse or identity scaled for singular case."""
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        if abs(det) < 1e-15:
            return [[1.0, 0.0], [0.0, 1.0]]
        return [
            [a[1][1] / det, -a[0][1] / det],
            [-a[1][0] / det, a[0][0] / det],
        ]

    # ═══════════════════════════════════════════════════════
    # Track Record Management
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def update_track_record(
        track_record: SourceTrackRecord,
        new_ic: Decimal,
        new_weight: Decimal | None = None,
        new_date: str | None = None,
    ) -> SourceTrackRecord:
        """追加新的观测到历史记录."""
        new_ics = list(track_record.ic_values) + [new_ic]
        new_weights = (list(track_record.weights) + [new_weight]
                       if track_record.weights and new_weight is not None
                       else None)
        new_dates = (list(track_record.dates) + [new_date]
                     if track_record.dates and new_date is not None
                     else None)
        return SourceTrackRecord(
            name=track_record.name,
            ic_values=new_ics,
            weights=new_weights,
            dates=new_dates,
        )
