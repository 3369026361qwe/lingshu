"""
信度理论引擎 — Bühlmann-Straub 信号融合 (v4.0).

纯计算层：无状态、无 IO。
将三路信号 (因子/GNN/Agent) 的线性 IC 加权升级为信度加权，
自动考虑每个信号源的时序稳定性和覆盖广度。

数学基础:
    Bühlmann-Straub 模型:
        μ = E[X_{i,t}]                         (总体均值)
        m = E[Var(X_{i,t} | Θ_i)]              (过程方差, within-source)
        v = Var(E[X_{i,t} | Θ_i])               (假设均值方差, between-source)
        Z_i = Σ w_{i,t} / (Σ w_{i,t} + m/v)    (信度因子)
        μ̂_i = Z_i · X̄_i^w + (1 - Z_i) · μ     (后验估计)

Usage:
    from jingsuan.credibility import CredibilityEngine, SourceTrackRecord
    weights = CredibilityEngine.buhlmann_straub(sources)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide, safe_mean


@dataclass
class SourceTrackRecord:
    """单个信号源的历史表现."""
    name: str
    ic_values: list[Decimal]
    weights: list[Decimal] | None = None  # 覆盖广度/成交额
    dates: list[str] | None = None

    @property
    def n_periods(self) -> int:
        """观测期数."""
        return len(self.ic_values)

    @property
    def mean_ic(self) -> Decimal:
        """历史 IC 均值."""
        return safe_mean(self.ic_values)

    @property
    def var_ic(self) -> Decimal:
        """历史 IC 方差."""
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


class CredibilityEngine:
    """信度理论引擎 — Bühlmann-Straub 多源融合。"""

    @staticmethod
    def buhlmann_straub(sources: list[SourceTrackRecord]) -> CredibilityWeights:
        """Bühlmann-Straub 信度模型.

        输入 3 个信号源的历史 IC 记录，输出信度加权融合权重。
        信号源 IC 越高且越稳定 → 信度因子越大 → 融合权重越大。
        """
        if len(sources) < 2:
            raise ValueError(f"需要 >= 2 个信号源, 当前 {len(sources)}")

        # 1. 估计总体均值 μ (三个源的平均 IC)
        all_ics = []
        for s in sources:
            all_ics.extend(s.ic_values)
        mu = safe_mean(all_ics) if all_ics else Decimal("0")

        # 2. 估计过程方差 m = E[Var(X_i | Θ_i)]
        #    每个源的内部方差平均值
        within_vars = []
        for s in sources:
            if s.n_periods < 2:
                continue
            mean_i = s.mean_ic
            var_i = sum((x - mean_i) ** 2 for x in s.ic_values) / Decimal(s.n_periods - 1)
            within_vars.append(var_i)
        m = safe_mean(within_vars) if within_vars else Decimal("1")

        # 3. 估计假设均值方差 v = Var(E[X_i | Θ_i])
        #    各源均值之间的方差
        source_means = [s.mean_ic for s in sources]
        mean_of_means = safe_mean(source_means)
        n_s = Decimal(len(sources))
        if n_s > 1:
            v = sum((sm - mean_of_means) ** 2 for sm in source_means) / (n_s - 1)
        else:
            v = Decimal("0")

        if v <= 0:
            v = m * Decimal("0.1")  # 最小方差比 m/v = 10 → Z ≈ 0.5

        # 4. 计算各源的信度因子 Z_i
        z_factors = {}
        weighted_means = {}

        for s in sources:
            if s.weights and len(s.weights) == s.n_periods:
                total_w = sum(s.weights)
                if total_w > 0:
                    weighted_mean = sum(
                        ic * w for ic, w in zip(s.ic_values, s.weights, strict=False)
                    ) / total_w
                else:
                    weighted_mean = s.mean_ic
            else:
                total_w = Decimal(s.n_periods)
                weighted_mean = s.mean_ic

            # Bühlmann 信度: Z = n / (n + m/v)
            n = Decimal(s.n_periods)
            z_i = safe_divide(n, n + safe_divide(m, v))
            z_factors[s.name] = z_i
            weighted_means[s.name] = weighted_mean

        # 5. 后验估计: μ̂_i = Z_i * X̄_i + (1 - Z_i) * μ
        posterior_ic = {}
        for s in sources:
            posterior_ic[s.name] = (
                z_factors[s.name] * weighted_means[s.name]
                + (Decimal("1") - z_factors[s.name]) * mu
            )

        # 6. 融合权重: softmax over posterior IC
        #    w_i = exp(μ̂_i / σ̂_i) / Σ exp(...), 其中 σ̂_i = sqrt(m / (n + m/v))
        posterior_vars = {}
        for s in sources:
            n = Decimal(s.n_periods)
            posterior_vars[s.name] = safe_divide(m, n + safe_divide(m, v))

        scores = {}
        for s in sources:
            sigma_hat = posterior_vars[s.name].sqrt()
            if sigma_hat > 0:
                scores[s.name] = (posterior_ic[s.name] / sigma_hat).exp()
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
    def fuse_signals(
        sources: list[SourceTrackRecord],
    ) -> list[Decimal]:
        """直接返回融合权重列表 [w_yinzi, w_gnn, w_agent]."""
        result = CredibilityEngine.buhlmann_straub(sources)
        return [
            result.source_weights.get(s.name, Decimal("0"))
            for s in sources
        ]
