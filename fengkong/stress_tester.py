"""历史场景压力测试器。"""
from decimal import Decimal


class StressTester:
    """历史极端场景压力测试。"""

    SCENARIOS = {
        "2008_financial_crisis": {"name": "2008金融危机", "market_shock": Decimal("-0.72"), "vol_multiplier": Decimal("3.0"), "liquidity_discount": Decimal("0.3")},
        "2015_market_crash": {"name": "2015股灾", "market_shock": Decimal("-0.45"), "vol_multiplier": Decimal("2.5"), "liquidity_discount": Decimal("0.2")},
        "2020_covid": {"name": "2020疫情冲击", "market_shock": Decimal("-0.15"), "vol_multiplier": Decimal("2.0"), "liquidity_discount": Decimal("0.1")},
        "2024_black_swan": {"name": "2024黑天鹅", "market_shock": Decimal("-0.30"), "vol_multiplier": Decimal("2.8"), "liquidity_discount": Decimal("0.25")},
    }

    @staticmethod
    def run_scenario(portfolio_value: Decimal, positions: dict[str, dict], scenario_name: str) -> dict:
        """运行单个压力测试场景。"""
        scenario = StressTester.SCENARIOS.get(scenario_name)
        if not scenario:
            return {"error": f"Unknown scenario: {scenario_name}"}

        shock = scenario["market_shock"]
        liq_discount = scenario["liquidity_discount"]
        total_loss = Decimal("0")

        for _code, pos in positions.items():
            mv = Decimal(str(pos.get("market_value", 0)))
            beta = Decimal(str(pos.get("beta", 1.0)))
            loss = mv * shock * beta
            loss += mv * liq_discount  # 流动性折扣
            total_loss += loss

        remaining = portfolio_value + total_loss
        return {"scenario": scenario["name"], "shock": shock, "estimated_loss": total_loss.quantize(Decimal("0.01")), "remaining_equity": remaining.quantize(Decimal("0.01")), "loss_ratio": (abs(total_loss) / portfolio_value).quantize(Decimal("0.0001")) if portfolio_value > 0 else Decimal("0")}

    @staticmethod
    def run_all(portfolio_value: Decimal, positions: dict[str, dict]) -> dict:
        """运行全部历史场景。"""
        results = {}
        for name in StressTester.SCENARIOS:
            results[name] = StressTester.run_scenario(portfolio_value, positions, name)
        return results
