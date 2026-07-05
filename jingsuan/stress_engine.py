"""
反向压力测试引擎 (v4.0).

纯计算层：给定组合和因子暴露，识别导致严重损失的市场情景。
补充现有的 4 个硬编码历史情景。

Usage:
    from jingsuan.stress_engine import StressEngine
    scenarios = StressEngine.factor_shock_scenarios(positions, factor_betas)
"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class StressScenario:
    name: str
    description: str
    shock_type: str       # "market" | "sector" | "liquidity" | "factor"
    loss_estimate: Decimal
    probability: Decimal  # 粗略概率估计


class StressEngine:
    """反向压力测试 + 结构化情景生成。"""

    HISTORICAL_SCENARIOS = {
        "2008_gfc": {"shock": Decimal("-0.72"), "vol_mult": Decimal("3"), "liquidity_discount": Decimal("0.30")},
        "2015_cn_crash": {"shock": Decimal("-0.45"), "vol_mult": Decimal("2"), "liquidity_discount": Decimal("0.25")},
        "2020_covid": {"shock": Decimal("-0.15"), "vol_mult": Decimal("1.5"), "liquidity_discount": Decimal("0.10")},
        "2024_blackswan": {"shock": Decimal("-0.30"), "vol_mult": Decimal("2"), "liquidity_discount": Decimal("0.20")},
    }

    @staticmethod
    def run_historical_scenarios(
        positions: list[tuple[str, Decimal, Decimal]],  # (code, weight, beta)
        market_value: Decimal,
    ) -> dict[str, Decimal]:
        """执行历史情景压力测试.

        Returns:
            {scenario_name: total_loss}
        """
        results = {}
        for name, params in StressEngine.HISTORICAL_SCENARIOS.items():
            shock = params["shock"]
            liq = params["liquidity_discount"]
            total_loss = Decimal("0")

            for _, weight, beta in positions:
                asset_loss = market_value * weight * shock * beta
                asset_loss += market_value * weight * liq
                total_loss += asset_loss

            results[name] = total_loss
        return results

    @staticmethod
    def factor_shock_scenarios(
        positions: list[tuple[str, Decimal, dict[str, Decimal]]],
        factor_names: list[str],
        shock_size: Decimal = Decimal("2"),  # 2 个标准差冲击
    ) -> list[StressScenario]:
        """基于因子暴露的反向压力测试.

        Args:
            positions: [(code, weight, {factor: beta})]
            factor_names: 因子名称列表
            shock_size: 标准差倍数
        Returns:
            每个因子的冲击情景
        """
        scenarios = []
        for factor in factor_names:
            total_impact = Decimal("0")
            for _, weight, betas in positions:
                beta = betas.get(factor, Decimal("0"))
                total_impact += weight * beta * shock_size

            prob = Decimal("0.05") if abs(total_impact) > Decimal("0.05") else Decimal("0.10")
            scenarios.append(StressScenario(
                name=f"factor_shock_{factor}",
                description=f"{factor} 因子 {shock_size}σ 冲击",
                shock_type="factor",
                loss_estimate=abs(total_impact),
                probability=prob,
            ))

        scenarios.sort(key=lambda s: float(s.loss_estimate), reverse=True)
        return scenarios

    @staticmethod
    def combined_stress(
        positions: list[tuple[str, Decimal, Decimal]],
        market_value: Decimal,
        market_shock: Decimal = Decimal("-0.10"),
        liquidity_shock: Decimal = Decimal("0.10"),
    ) -> StressScenario:
        """复合压力测试: 市场冲击 + 流动性冲击.

        Returns:
            单情景: 市场大跌 + 流动性紧缩
        """
        total = Decimal("0")
        for _, weight, beta in positions:
            market_loss = market_value * weight * market_shock * beta
            liq_loss = market_value * weight * liquidity_shock
            total += market_loss + liq_loss

        return StressScenario(
            name="combined_market_liquidity",
            description=f"市场 {market_shock} + 流动性 {liquidity_shock}",
            shock_type="market",
            loss_estimate=abs(total),
            probability=Decimal("0.05"),
        )
