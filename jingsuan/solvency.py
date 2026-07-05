"""
Solvency II SCR 计算器 — 风险资本聚合 (v4.0).

纯计算层：无状态、无 IO。
将 EVT VaR 转换为 Solvency II 标准公式下的风险资本要求，
提供风险分解和风险预算分配。

数学基础:
    SCR_total = sqrt( Σ SCR_i² + Σ 2ρ_ij·SCR_i·SCR_j )
    标准相关矩阵: ρ(market,credit)=0.25, ρ(market,op)=0, ρ(credit,op)=0

Usage:
    from jingsuan.solvency import SCRCalculator
    scr = SCRCalculator.calculate_scr(evt_var_995, ...)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class SCRDecomposition:
    scr_total: Decimal
    scr_equity: Decimal
    scr_concentration: Decimal
    scr_credit: Decimal
    scr_operational: Decimal
    diversification_benefit: Decimal
    risk_budgets: dict[str, Decimal]


# EIOPA 标准公式相关矩阵 (简化)
_CORR_MATRIX = {
    ("equity", "concentration"): Decimal("0.5"),
    ("equity", "credit"): Decimal("0.25"),
    ("equity", "operational"): Decimal("0"),
    ("concentration", "credit"): Decimal("0.25"),
    ("concentration", "operational"): Decimal("0"),
    ("credit", "operational"): Decimal("0"),
}


class SCRCalculator:
    """Solvency II SCR 计算器. 纯静态方法, 无 IO."""

    @staticmethod
    def calculate_scr(
        evt_var_995: Decimal,                    # EVT 99.5% VaR
        position_concentration: Decimal,          # Herfindahl 指数
        aum: Decimal,                             # 管理资产规模
        credit_spread_risk: Decimal | None = None,
    ) -> SCRDecomposition:
        """计算总 SCR 并按风险模块分解.

        Args:
            evt_var_995: EVT引擎的 99.5% VaR
            position_concentration: 集中度风险度量 (Herfindahl)
            aum: 管理资产规模
            credit_spread_risk: 信用价差风险, 默认 = 0.02 * AUM
        """
        # Market risk sub-modules
        scr_equity = evt_var_995
        scr_concentration = position_concentration * aum
        scr_credit = (credit_spread_risk if credit_spread_risk is not None
                      else aum * Decimal("0.02"))
        scr_operational = aum * Decimal("0.03")  # 简化: 3% AUM

        risks = {
            "equity": scr_equity,
            "concentration": scr_concentration,
            "credit": scr_credit,
            "operational": scr_operational,
        }

        # 聚合
        scr_sq = sum(v * v for v in risks.values())
        for (r1, r2), rho in _CORR_MATRIX.items():
            scr_sq += Decimal("2") * rho * risks[r1] * risks[r2]

        scr_total = scr_sq.sqrt()

        # 分散化收益
        sum_standalone = sum(risks.values())
        div_benefit = Decimal("1") - safe_divide(scr_total, sum_standalone) if sum_standalone > 0 else Decimal("0")

        # 风险预算
        risk_budgets = {
            name: safe_divide(val, scr_total) if scr_total > 0 else Decimal("0")
            for name, val in risks.items()
        }

        return SCRDecomposition(
            scr_total=scr_total,
            scr_equity=scr_equity,
            scr_concentration=scr_concentration,
            scr_credit=scr_credit,
            scr_operational=scr_operational,
            diversification_benefit=div_benefit,
            risk_budgets=risk_budgets,
        )

    @staticmethod
    def risk_budget_to_position_limits(
        scr_decomp: SCRDecomposition,
        max_position: Decimal = Decimal("0.10"),
    ) -> dict[str, Decimal]:
        """将 SCR 风险预算转化为仓位约束.

        权益风险占比 * 单票上限 → 调整后的动态上限.
        """
        eq_budget = scr_decomp.risk_budgets.get("equity", Decimal("1"))
        concentration_budget = scr_decomp.risk_budgets.get("concentration", Decimal("0"))

        return {
            "single_stock_max": max_position * eq_budget,
            "single_industry_max": max_position * Decimal("3") * concentration_budget,
        }
