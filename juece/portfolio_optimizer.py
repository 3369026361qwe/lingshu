"""
Black-Litterman 组合优化器。

结合市场均衡收益 + 主观观点（Agent信号），输出最优权重。
"""

from decimal import Decimal


class PortfolioOptimizer:
    """组合优化器 — 基于 Black-Litterman 框架的简化实现。

    核心公式:
        Π = δ Σ w_mkt          (均衡收益)
        E(R) = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P^T Ω^{-1} Q]
    """

    def __init__(
        self,
        risk_aversion: Decimal = Decimal("3.0"),
        tau: Decimal = Decimal("0.05"),
        max_weight: Decimal = Decimal("0.10"),
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.max_weight = max_weight

    # ── 优化 ────────────────────────────────────────────

    def optimize(
        self,
        picks: list[dict],          # [{code, score, weight}]
        views: dict[str, Decimal] | None = None,  # {code: expected_return}
        confidence: Decimal = Decimal("0.5"),
    ) -> list[dict]:
        """简化 Black-Litterman 优化。

        Args:
            picks: 候选股票列表
            views: 主观观点（Agent 预测收益），None 则使用 score 作为隐含收益
            confidence: 观点置信度 [0, 1]

        Returns:
            [{code, weight, expected_return}]
        """
        if not picks:
            return []

        n = len(picks)
        codes = [p["code"] for p in picks]

        # 先验权重（等权）
        prior_weights = [Decimal("1") / Decimal(str(n)) for _ in range(n)]

        # 观点收益（从 score 推导，或使用外部 views）
        if views:
            view_returns = [views.get(c, Decimal("0")) for c in codes]
        else:
            max_score = max(p["score"] for p in picks)
            min_score = min(p["score"] for p in picks)
            denom = max_score - min_score if max_score != min_score else Decimal("1")
            view_returns = [(p["score"] - min_score) / denom * Decimal("0.10") for p in picks]

        # BL 融合: 后验 = (1 - confidence) × 先验 + confidence × 观点
        posterior = [
            (Decimal("1") - confidence) * prior_weights[i] + confidence * view_returns[i]
            for i in range(n)
        ]

        # 归一化并裁剪
        total = sum(max(Decimal("0"), w) for w in posterior)
        if total == 0:
            total = Decimal("1")

        result = []
        for i, code in enumerate(codes):
            raw_weight = max(Decimal("0"), posterior[i]) / total
            weight = min(raw_weight, self.max_weight)
            result.append({
                "code": code,
                "weight": weight.quantize(Decimal("0.000001")),
                "expected_return": view_returns[i].quantize(Decimal("0.0001")),
            })

        # 重新归一化
        total_w = sum(r["weight"] for r in result)
        if total_w > 0:
            for r in result:
                r["weight"] = (r["weight"] / total_w).quantize(Decimal("0.000001"))

        return result

    # ── 约束检查 ────────────────────────────────────────

    def check_constraints(
        self,
        portfolio: list[dict],
        max_single: Decimal = Decimal("0.10"),
        max_industry: Decimal = Decimal("0.30"),
        total_max: Decimal = Decimal("0.95"),
    ) -> list[str]:
        """检查组合是否满足约束条件。

        Returns:
            违规项列表
        """
        violations = []
        total = sum(r["weight"] for r in portfolio)

        if total > total_max:
            violations.append(f"总仓位 {float(total):.1%} > {float(total_max):.0%}")

        for r in portfolio:
            if r["weight"] > max_single:
                violations.append(f"{r['code']} 单票 {float(r['weight']):.1%} > {float(max_single):.0%}")

        return violations
