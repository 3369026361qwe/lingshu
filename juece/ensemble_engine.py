"""
三路信号加权融合引擎 (v4.0 升级: Bühlmann-Straub 信度融合).

支持两种模式:
    1. 传统 IC 驱动 (update_weights_from_ic) — 向后兼容
    2. 信度融合 (update_weights_credibility) — v4.0 新增, 更稳健

最终得分 = w1 × 多因子得分 + w2 × GNN增强得分 + w3 × Agent综合评分
"""

from decimal import Decimal

from jingsuan.credibility import CredibilityEngine, SourceTrackRecord


class EnsembleEngine:
    """三路信号加权融合引擎."""

    def __init__(self):
        self._weights = {
            "factor": Decimal("0.40"),
            "gnn": Decimal("0.30"),
            "agent": Decimal("0.30"),
        }
        # 历史 IC 记录 (用于信度融合)
        self._track_records: dict[str, list[Decimal]] = {
            "factor": [], "gnn": [], "agent": [],
        }

    @property
    def weights(self) -> dict[str, Decimal]:
        return dict(self._weights)

    # ── 传统 IC 驱动 (向后兼容) ───────────────────────────

    def update_weights_from_ic(
        self,
        factor_ic: Decimal | None = None,
        gnn_ic: Decimal | None = None,
        agent_ic: Decimal | None = None,
    ) -> dict[str, Decimal]:
        """根据各信号源的历史 IC 表现动态调整融合权重."""
        ics = {}
        if factor_ic is not None and factor_ic > 0:
            ics["factor"] = factor_ic
        if gnn_ic is not None and gnn_ic > 0:
            ics["gnn"] = gnn_ic
        if agent_ic is not None and agent_ic > 0:
            ics["agent"] = agent_ic

        if not ics:
            return self.weights

        total_ic = sum(ics.values())
        new_weights = {}
        for key in self._weights:
            new_weights[key] = ics[key] / total_ic if key in ics else Decimal("0")

        self._weights.update(new_weights)
        return self.weights

    # ── 信度融合 (v4.0 新增) ──────────────────────────────

    def update_weights_credibility(self) -> dict[str, Decimal]:
        """Bühlmann-Straub 信度融合 — 自动考虑时序稳定性."""
        sources = [
            SourceTrackRecord(name, ics) for name, ics in self._track_records.items()
        ]
        try:
            result = CredibilityEngine.buhlmann_straub(sources)
            self._weights = dict(result.source_weights)
        except (ValueError, Exception):
            pass
        return self.weights

    def record_ic(
        self,
        factor_ic: Decimal | None = None,
        gnn_ic: Decimal | None = None,
        agent_ic: Decimal | None = None,
    ) -> None:
        """记录各源 IC 用于信度估计."""
        if factor_ic is not None:
            self._track_records["factor"].append(factor_ic)
        if gnn_ic is not None:
            self._track_records["gnn"].append(gnn_ic)
        if agent_ic is not None:
            self._track_records["agent"].append(agent_ic)

    def set_weights(self, factor: Decimal, gnn: Decimal, agent: Decimal) -> None:
        total = factor + gnn + agent
        self._weights["factor"] = factor / total
        self._weights["gnn"] = gnn / total
        self._weights["agent"] = agent / total

    # ── 融合计算 ────────────────────────────────────────

    def fuse(
        self,
        factor_scores: dict[str, Decimal],       # {code: factor_score}
        gnn_scores: dict[str, float] | None = None,   # {code: gnn_score}
        agent_scores: dict[str, Decimal] | None = None, # {code: agent_score}
        normalize: bool = True,
    ) -> dict[str, Decimal]:
        """三路信号加权融合 → 全市场综合得分。

        Args:
            factor_scores: 多因子得分
            gnn_scores: GNN 增强得分
            agent_scores: Agent 综合评分
            normalize: 是否对每路信号先做 min-max 归一化

        Returns:
            {code: composite_score}
        """
        # 收集所有股票代码
        all_codes = set(factor_scores.keys())
        if gnn_scores:
            all_codes.update(gnn_scores.keys())
        if agent_scores:
            all_codes.update(agent_scores.keys())

        # 归一化各路信号到 [0, 1]
        f_norm = self._minmax_norm(factor_scores) if normalize else factor_scores
        g_norm = self._minmax_norm(gnn_scores) if normalize and gnn_scores else (gnn_scores or {})
        a_norm = self._minmax_norm(agent_scores) if normalize and agent_scores else (agent_scores or {})

        w_f = self._weights["factor"]
        w_g = self._weights["gnn"]
        w_a = self._weights["agent"]

        composite = {}
        for code in all_codes:
            score = Decimal("0")
            if code in f_norm:
                score += w_f * Decimal(str(f_norm[code]))
            if code in g_norm:
                score += w_g * Decimal(str(g_norm[code]))
            if code in a_norm:
                score += w_a * Decimal(str(a_norm[code]))
            composite[code] = score.quantize(Decimal("0.0001"))

        return composite

    def rank(self, composite: dict[str, Decimal]) -> list[tuple[str, Decimal]]:
        """按综合得分降序排序。"""
        return sorted(composite.items(), key=lambda x: x[1], reverse=True)

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _minmax_norm(scores: dict | None) -> dict:
        """Min-Max 归一化到 [0, 1]。返回 float 值。"""
        if not scores:
            return {}
        vals = list(scores.values())
        v_min = min(vals)
        v_max = max(vals)
        if v_max == v_min:
            return {k: 0.5 for k in scores}
        denom = v_max - v_min
        return {
            k: float((v - v_min) / denom) if not isinstance(v, (int, float)) else (v - v_min) / denom
            for k, v in scores.items()
        }

    @staticmethod
    def _zscore_norm(scores: dict[str, Decimal]) -> dict[str, Decimal]:
        """Z-Score 标准化。"""
        if not scores:
            return {}
        vals = list(scores.values())
        n = Decimal(len(vals))
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var.sqrt()
        if std == 0:
            return {k: Decimal("0") for k in scores}
        return {k: (v - mean) / std for k, v in scores.items()}
