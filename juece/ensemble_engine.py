"""
三路信号加权融合引擎。

最终得分 = w1 × 多因子得分 + w2 × GNN增强得分 + w3 × Agent综合评分

权重 w1, w2, w3 由历史 IC 表现动态调整。
"""

from decimal import Decimal
from typing import Optional


class EnsembleEngine:
    """三路信号加权融合引擎。"""

    def __init__(self):
        # 默认权重（等权启动，后续由 IC 动态调整）
        self._weights = {
            "factor": Decimal("0.40"),
            "gnn": Decimal("0.30"),
            "agent": Decimal("0.30"),
        }

    @property
    def weights(self) -> dict[str, Decimal]:
        return dict(self._weights)

    # ── 权重调整 ────────────────────────────────────────

    def update_weights_from_ic(
        self,
        factor_ic: Optional[Decimal] = None,
        gnn_ic: Optional[Decimal] = None,
        agent_ic: Optional[Decimal] = None,
    ) -> dict[str, Decimal]:
        """根据各信号源的历史 IC 表现动态调整融合权重。

        IC 越高 → 权重越大（等比例分配）。
        """
        ics = {}
        if factor_ic is not None and factor_ic > 0:
            ics["factor"] = factor_ic
        if gnn_ic is not None and gnn_ic > 0:
            ics["gnn"] = gnn_ic
        if agent_ic is not None and agent_ic > 0:
            ics["agent"] = agent_ic

        if not ics:
            return self.weights  # 无有效 IC，保持当前权重

        total_ic = sum(ics.values())
        new_weights = {}
        for key in self._weights:
            if key in ics:
                new_weights[key] = ics[key] / total_ic
            else:
                new_weights[key] = Decimal("0")

        self._weights.update(new_weights)
        return self.weights

    def set_weights(self, factor: Decimal, gnn: Decimal, agent: Decimal) -> None:
        total = factor + gnn + agent
        self._weights["factor"] = factor / total
        self._weights["gnn"] = gnn / total
        self._weights["agent"] = agent / total

    # ── 融合计算 ────────────────────────────────────────

    def fuse(
        self,
        factor_scores: dict[str, Decimal],       # {code: factor_score}
        gnn_scores: Optional[dict[str, float]] = None,   # {code: gnn_score}
        agent_scores: Optional[dict[str, Decimal]] = None, # {code: agent_score}
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
    def _minmax_norm(scores: Optional[dict]) -> dict:
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
