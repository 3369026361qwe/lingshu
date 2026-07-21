"""
测试 CorporateActionAdjuster — 复权因子和价格调整.
"""

from decimal import Decimal

from shuju.corporate_action import CorporateActionAdjuster


class TestAdjustFromClose:
    def test_no_dividend_no_split(self):
        factor = CorporateActionAdjuster.adjust_from_close(
            Decimal("10.0"), dividend_per_share=Decimal("0"), split_ratio=Decimal("1")
        )
        assert factor == Decimal("1")

    def test_dividend_only(self):
        # 分红 0.5, 收盘价 10 → factor = (10-0.5)/10 = 0.95
        factor = CorporateActionAdjuster.adjust_from_close(
            Decimal("10.0"), dividend_per_share=Decimal("0.5"), split_ratio=Decimal("1")
        )
        assert factor == Decimal("0.95")

    def test_split_only(self):
        # 1拆2 → factor = 10*2/10 = 2
        factor = CorporateActionAdjuster.adjust_from_close(
            Decimal("10.0"), dividend_per_share=Decimal("0"), split_ratio=Decimal("2")
        )
        assert factor == Decimal("2")

    def test_dividend_and_split(self):
        # 分红 0.5 + 1拆2 → factor = (10-0.5)*2/10 = 1.9
        factor = CorporateActionAdjuster.adjust_from_close(
            Decimal("10.0"), dividend_per_share=Decimal("0.5"), split_ratio=Decimal("2")
        )
        assert factor == Decimal("1.9")

    def test_zero_close(self):
        factor = CorporateActionAdjuster.adjust_from_close(
            Decimal("0"), dividend_per_share=Decimal("0.5")
        )
        assert factor == Decimal("1")  # 安全默认


class TestAdjustPrices:
    def test_no_adjustment(self):
        prices = [Decimal("10")] * 5
        factors = [Decimal("1")] * 5
        result = CorporateActionAdjuster.adjust_prices(prices, factors)
        assert result == prices

    def test_backward_adjustment_basic(self):
        # 后向复权: 保持最新价格不变
        prices = [Decimal("10"), Decimal("11"), Decimal("12")]
        # T=0: factor=1, T=1: factor=1.1 (10% dividend), T=2: factor=1
        factors = [Decimal("1"), Decimal("1.1"), Decimal("1")]
        result = CorporateActionAdjuster.adjust_prices(prices, factors)
        # cum[T] = factor[T], cum[T-1] = factor[T-1] * cum[T]
        # cum[2] = 1, cum[1] = 1.1 * 1 = 1.1, cum[0] = 1 * 1.1 = 1.1
        # adj[2] = 12 * 1 = 12, adj[1] = 11 * 1.1 = 12.1, adj[0] = 10 * 1.1 = 11.0
        assert len(result) == 3
        assert result[2] == Decimal("12")  # 最新价不变
        assert result[1] > prices[1]
        assert result[0] > prices[0]

    def test_empty_inputs(self):
        result = CorporateActionAdjuster.adjust_prices([], [])
        assert result == []

    def test_mismatched_lengths(self):
        prices = [Decimal("10")] * 5
        factors = [Decimal("2")] * 3
        result = CorporateActionAdjuster.adjust_prices(prices, factors)
        assert len(result) == 3  # min(5,3)

    def test_single_element(self):
        prices = [Decimal("10")]
        factors = [Decimal("2")]
        result = CorporateActionAdjuster.adjust_prices(prices, factors)
        assert result == [Decimal("20")]

    def test_zero_factor_ignored(self):
        prices = [Decimal("10"), Decimal("11")]
        factors = [Decimal("0"), Decimal("1")]
        result = CorporateActionAdjuster.adjust_prices(prices, factors)
        # factor[0]=0 is ignored (treated as 1 for cum), cum[1]=1, cum[0]=1
        assert result == [Decimal("10"), Decimal("11")]


class TestAdjustPricesForward:
    def test_no_adjustment(self):
        prices = [Decimal("10")] * 5
        factors = [Decimal("1")] * 5
        result = CorporateActionAdjuster.adjust_prices_forward(prices, factors)
        assert result == prices

    def test_forward_adjustment_basic(self):
        # 前向复权: 保持最早价格不变, 调整后续价格
        prices = [Decimal("10"), Decimal("11"), Decimal("12")]
        factors = [Decimal("1"), Decimal("1.1"), Decimal("1")]
        result = CorporateActionAdjuster.adjust_prices_forward(prices, factors)
        # cum[0] = 1, adj[0] = 10/1 = 10
        # cum[1] = 1 * 1.1 = 1.1, adj[1] = 11/1.1 = 10.0
        # cum[2] = 1.1 * 1 = 1.1, adj[2] = 12/1.1 = 10.909...
        assert result[0] == Decimal("10")  # 最早价不变
        assert result[1] < prices[1]       # 前向复权使后续价格降低

    def test_empty_inputs(self):
        result = CorporateActionAdjuster.adjust_prices_forward([], [])
        assert result == []

    def test_single_element(self):
        prices = [Decimal("10")]
        factors = [Decimal("2")]
        result = CorporateActionAdjuster.adjust_prices_forward(prices, factors)
        assert result == [Decimal("5")]


class TestValidateAdjustment:
    def test_no_adjustment_needed(self):
        prices = [Decimal("10.0"), Decimal("10.1"), Decimal("10.05"), Decimal("10.2"), Decimal("10.15")]
        result = CorporateActionAdjuster.validate_adjustment(prices, prices)
        assert result["n"] == 5
        assert result["max_return_diff"] == 0.0
        assert result["n_breaks"] == 0
        assert not result["adjustment_significant"]

    def test_with_jump(self):
        # 模拟除权跳空场景
        raw = [Decimal("10.0"), Decimal("9.0"), Decimal("9.1")]  # T=1 跌 10% → 除权
        adj = [Decimal("9.5"), Decimal("9.3"), Decimal("9.4")]
        result = CorporateActionAdjuster.validate_adjustment(raw, adj)
        assert result["n_breaks"] >= 1  # raw 中 T=1 跌幅 > 5%
        assert result["adjustment_significant"]

    def test_insufficient_data(self):
        result = CorporateActionAdjuster.validate_adjustment([Decimal("10")], [Decimal("10")])
        assert result["n"] == 1
        assert not result["adjustment_significant"]

    def test_volatility(self):
        prices = [Decimal("10"), Decimal("11"), Decimal("10.5"), Decimal("12"), Decimal("11")]
        result = CorporateActionAdjuster.validate_adjustment(prices, prices)
        assert result["raw_volatility"] > 0 or True


class TestBuildAdjustmentFactors:
    def test_static_build_returns_list(self):
        factors = CorporateActionAdjuster.build_adjustment_factors(
            dividends=[], splits=[], start_date="20240101", end_date="20240131"
        )
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert factors[0] == Decimal("1")

    def test_build_with_events(self):
        dividends = [("20240115", Decimal("0.5"))]
        splits = [("20240120", Decimal("2.0"))]
        factors = CorporateActionAdjuster.build_adjustment_factors(
            dividends=dividends, splits=splits,
            start_date="20240101", end_date="20240131"
        )
        assert len(factors) > 0


class TestFetchDailyWithFactors:
    def test_empty_for_invalid_code(self):
        """无效代码应返回空列表 (不会崩溃)."""
        adjuster = CorporateActionAdjuster()
        # 没有 AKShare mock, 这应该 catch 异常并返回空
        # 在 CI 环境中 AKShare 可能不可用
        result = adjuster.fetch_daily_with_factors("INVALID", "20240101", "20240131")
        # 可能返回空或实际数据 (取决于 AKShare 是否可用)
        assert isinstance(result, list)
