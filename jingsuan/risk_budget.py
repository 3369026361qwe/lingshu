"""
动态风险预算引擎 (v4.0).

纯计算层：输入 EVT VaR + SCR + 破产分析，输出每只股票的头寸上限。
替代固定 10%/30%/95% 限制。

Usage:
    from jingsuan import RiskBudgetEngine
    limits = RiskBudgetEngine.compute_limits(evt_var, ruin, scr, positions)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class RiskLimits:
    single_stock_max: Decimal
    single_industry_max: Decimal
    total_exposure_max: Decimal
    var_limit: Decimal
    explanation: str


class RiskBudgetEngine:
    """动态风险预算 — 基于精算引擎的多维仓位限制。"""

    @staticmethod
    def compute_limits(
        evt_var_99: Decimal,              # EVT 99% VaR (比例)
        optimal_f: Decimal,               # 破产理论最优仓位
        scr_decomp,                       # SCRDecomposition
        n_positions: int = 30,
    ) -> RiskLimits:
        """综合 EVT / Ruin / SCR 三维度计算动态仓位限制.

        Args:
            evt_var_99: EVT 99% VaR (比例, 如 0.08 = 8%)
            optimal_f: 破产理论最优仓位 (比例)
            scr_decomp: SCR 分解结果
            n_positions: 组合中持仓数量
        Returns:
            RiskLimits
        """
        # 1. EVT 维度: 尾部风险越大 → 单票上限越小
        evt_limit = safe_divide(Decimal("0.15"), evt_var_99 * Decimal("2"))
        evt_limit = max(Decimal("0.02"), min(evt_limit, Decimal("0.15")))

        # 2. Ruin 维度: 破产约束仓位
        ruin_limit = optimal_f

        # 3. SCR 维度: 风险预算分配
        eq_budget = scr_decomp.risk_budgets.get("equity", Decimal("1"))
        scr_limit = Decimal("0.15") * eq_budget

        # 综合: 取三维度的最小值 (最保守)
        single_stock = min(evt_limit, ruin_limit, scr_limit)
        single_stock = max(Decimal("0.01"), single_stock)  # 最小 1% (允许小额试探)

        # 行业限制 = 单个限制 * 3 (不能一行业压太重)
        industry_max = min(single_stock * Decimal("3"), Decimal("0.30"))

        # 总敞口 = min(单个 * N, 破产约束 * N, 95%)
        total_exposure = min(single_stock * Decimal(n_positions), Decimal("0.95"))

        # VaR 限制: 组合 VaR 不能超过 EVT VaR * 2
        var_limit = evt_var_99 * Decimal("2")

        explanation = (
            f"EVT_limit={evt_limit:.4f} Ruin_limit={ruin_limit:.4f} "
            f"SCR_limit={scr_limit:.4f} → single={single_stock:.4f}"
        )

        return RiskLimits(
            single_stock_max=single_stock,
            single_industry_max=industry_max,
            total_exposure_max=total_exposure,
            var_limit=var_limit,
            explanation=explanation,
        )

    @staticmethod
    def marginal_var_contribution(
        weights: list[Decimal],
        cov_matrix: list[list[Decimal]],
        confidence: Decimal = Decimal("0.99"),
    ) -> list[Decimal]:
        """边际 VaR 贡献: MVaR_i = ∂VaR/∂w_i ∝ (Σ·w)_i / sqrt(w'Σw).

        Args:
            weights: 组合权重
            cov_matrix: 协方差矩阵 [i][j]
            confidence: 置信水平
        Returns:
            各资产边际 VaR 贡献列表
        """
        n = len(weights)
        z_score = Decimal("2.326") if confidence >= Decimal("0.99") else Decimal("1.645")

        # 组合方差 = w'Σw
        port_var = Decimal("0")
        for i in range(n):
            for j in range(n):
                port_var += weights[i] * cov_matrix[i][j] * weights[j]

        port_sigma = port_var.sqrt()
        if port_sigma == 0:
            return [Decimal("0")] * n

        # 边际贡献 ∝ (Σ·w)_i
        mvar = []
        for i in range(n):
            cov_w_i = Decimal("0")
            for j in range(n):
                cov_w_i += cov_matrix[i][j] * weights[j]
            # MVaR_i = z * cov_w_i / port_sigma
            mvar.append(z_score * safe_divide(cov_w_i, port_sigma))

        return mvar
