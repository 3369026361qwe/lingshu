"""仓位限制器 — L2 防护层。单票≤10% | 单行业≤30% | 总仓位≤95%。"""
from decimal import Decimal


class PositionLimiter:
    """仓位/行业/单票限制器。"""

    def __init__(self, max_single: Decimal = Decimal("0.10"), max_industry: Decimal = Decimal("0.30"), max_total: Decimal = Decimal("0.95")):
        self.max_single = max_single
        self.max_industry = max_industry
        self.max_total = max_total

    def check(self, portfolio: list[dict], industry_map: dict[str, str] | None = None) -> dict:
        """检查组合是否合规。

        Returns:
            {passed: bool, violations: [str], advice: str}
        """
        violations = []
        total = sum(Decimal(str(r.get("weight", 0))) for r in portfolio)

        if total > self.max_total:
            violations.append(f"总仓位 {float(total):.1%} > {float(self.max_total):.0%}")

        for r in portfolio:
            w = Decimal(str(r.get("weight", 0)))
            if w > self.max_single:
                violations.append(f"{r['code']} 单票 {float(w):.1%} > {float(self.max_single):.0%}")

        if industry_map:
            ind_weights: dict[str, Decimal] = {}
            for r in portfolio:
                ind = industry_map.get(r["code"], "未知")
                ind_weights[ind] = ind_weights.get(ind, Decimal("0")) + Decimal(str(r.get("weight", 0)))
            for ind, w in ind_weights.items():
                if w > self.max_industry:
                    violations.append(f"{ind} 行业 {float(w):.1%} > {float(self.max_industry):.0%}")

        return {"passed": len(violations) == 0, "violations": violations, "advice": "合规" if not violations else f"{len(violations)}项违规"}

    def calc_kelly(self, win_rate: Decimal, avg_win: Decimal, avg_loss: Decimal) -> Decimal | None:
        """凯利公式: f = p - q / (W/L)。"""
        if avg_loss == 0:
            return None
        p = win_rate
        q = Decimal("1") - p
        ratio = avg_win / avg_loss
        f = p - q / ratio
        return max(Decimal("0"), min(f, self.max_single))
