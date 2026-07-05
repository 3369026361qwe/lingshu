"""
归因分析引擎 (v4.0).

三维归因: Brinson (配置+选择+交互) + 因子 + 风险.
回答"钱从哪来"的尽调核心问题.

Usage:
    from huice.attribution import AttributionEngine
    result = AttributionEngine.brinson(portfolio_weights, benchmark_weights, ...)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class BrinsonAttribution:
    allocation_effect: list[Decimal]
    selection_effect: list[Decimal]
    interaction_effect: list[Decimal]
    total_active_return: Decimal
    sectors: list[str]


@dataclass
class FactorAttribution:
    factor_contributions: dict[str, Decimal]
    alpha: Decimal  # 未被因子解释的收益 (= 经理技能)
    r_squared: Decimal


@dataclass
class RiskAttribution:
    marginal_var: list[Decimal]
    component_var: list[Decimal]
    var_total: Decimal


class AttributionEngine:
    """归因分析 — Brinson + 因子 + 风险."""

    @staticmethod
    def brinson(
        portfolio_weights: dict[str, list[tuple[str, Decimal]]],
        benchmark_weights: dict[str, list[tuple[str, Decimal]]],
        portfolio_returns: dict[str, Decimal],
        benchmark_returns: dict[str, Decimal],
    ) -> BrinsonAttribution:
        """Brinson 归因.

        Allocation  = (w_p,s - w_b,s) * (r_b,s - r_b)
        Selection   = w_b,s * (r_p,s - r_b,s)
        Interaction = (w_p,s - w_b,s) * (r_p,s - r_b,s)
        """
        # 计算总体基准回报
        total_b_return = Decimal("0")
        for _, assets in benchmark_weights.items():
            for code, w in assets:
                if code in benchmark_returns:
                    total_b_return += w * benchmark_returns[code]

        sectors = []
        alloc = []
        sel = []
        inter = []

        all_sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        for sector in all_sectors:
            pw = portfolio_weights.get(sector, [])
            bw = benchmark_weights.get(sector, [])

            # 行业权重
            w_p_s = sum(w for _, w in pw)
            w_b_s = sum(w for _, w in bw)

            # 行业组合回报
            r_p_s = AttributionEngine._sector_return(pw, portfolio_returns)
            r_b_s = AttributionEngine._sector_return(bw, benchmark_returns)

            # Brinson 三效应
            alloc_s = (w_p_s - w_b_s) * (r_b_s - total_b_return)
            sel_s = w_b_s * (r_p_s - r_b_s) if w_b_s > 0 else Decimal("0")
            inter_s = (w_p_s - w_b_s) * (r_p_s - r_b_s)

            sectors.append(sector)
            alloc.append(alloc_s)
            sel.append(sel_s)
            inter.append(inter_s)

        total_active = sum(alloc) + sum(sel) + sum(inter)

        return BrinsonAttribution(
            allocation_effect=alloc, selection_effect=sel,
            interaction_effect=inter, total_active_return=total_active,
            sectors=sectors,
        )

    @staticmethod
    def factor_attribution(
        portfolio_returns: list[Decimal],
        factor_returns: dict[str, list[Decimal]],
        factor_exposures: dict[str, list[Decimal]],
    ) -> FactorAttribution:
        """因子归因: R_p = Σ β_k * F_k + α."""
        n = len(portfolio_returns)
        contributions = {}
        for name, f_returns in factor_returns.items():
            beta = safe_divide(sum(factor_exposures.get(name, [Decimal("0")] * n)[i]
                                   for i in range(n)) if n > 0 else Decimal("0"),
                               Decimal(n))
            factor_contrib = beta * safe_divide(sum(f_returns), Decimal(max(1, n)))
            contributions[name] = factor_contrib

        # Alpha = 总收益 - 因子解释收益
        total_ret = sum(portfolio_returns) / Decimal(max(1, n))
        explained = sum(contributions.values())
        alpha = total_ret - explained

        # R² 近似
        ss_total = sum((r - total_ret) ** 2 for r in portfolio_returns) / Decimal(max(1, n))
        r2 = safe_divide(explained * explained, ss_total) if ss_total > 0 else Decimal("0")

        return FactorAttribution(
            factor_contributions=contributions, alpha=alpha, r_squared=r2,
        )

    @staticmethod
    def risk_attribution(
        weights: list[Decimal],
        cov_matrix: list[list[Decimal]],
        confidence: Decimal = Decimal("0.99"),
    ) -> RiskAttribution:
        """风险归因: 边际 VaR + 成分 VaR.

        MVaR_i = ∂VaR/∂w_i = z * (Σ·w)_i / sqrt(w'Σw)
        CVaR_i = w_i * MVaR_i
        """
        n = len(weights)
        z = Decimal("2.326") if confidence >= Decimal("0.99") else Decimal("1.645")

        port_var = Decimal("0")
        for i in range(n):
            for j in range(n):
                port_var += weights[i] * cov_matrix[i][j] * weights[j]
        port_vol = port_var.sqrt()

        if port_vol == 0:
            return RiskAttribution(
                marginal_var=[Decimal("0")] * n,
                component_var=[Decimal("0")] * n,
                var_total=Decimal("0"),
            )

        mvar = []
        cvar = []
        for i in range(n):
            cov_w = sum(cov_matrix[i][j] * weights[j] for j in range(n))
            mv = z * safe_divide(cov_w, port_vol)
            mvar.append(mv)
            cvar.append(weights[i] * mv)

        var_total = z * port_vol
        return RiskAttribution(marginal_var=mvar, component_var=cvar, var_total=var_total)

    @staticmethod
    def _sector_return(assets, returns_dict):
        total_w = sum(w for _, w in assets)
        if total_w == 0:
            return Decimal("0")
        return sum(w * returns_dict.get(code, Decimal("0")) for code, w in assets) / total_w
