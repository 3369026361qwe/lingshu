"""
测试因子计算: 估值/动量/质量/波动率/情绪/另类。
"""

from decimal import Decimal

import pytest

from yinzi.value_factors import PEFactor, PBFactor, PSFactor, FCFYieldFactor, PEGFactor
from yinzi.momentum_factors import (
    Momentum1MFactor, Momentum3MFactor, Momentum6MFactor, Momentum12M1MFactor, TurnoverMomentumFactor,
)
from yinzi.quality_factors import (
    ROEFactor, ROAFactor, GrossMarginFactor, NetMarginFactor, CashflowToRevenueFactor,
)
from yinzi.volatility_factors import (
    HistoricalVolFactor, DownsideVolFactor, BetaFactor, VaRFactor,
)
from yinzi.sentiment_factors import (
    VolumeRatioFactor, MoneyFlowFactor, TurnoverAnomalyFactor, NorthBoundFactor,
)
from yinzi.alternative_factors import (
    AnalystCoverageFactor, InstitutionalHoldingFactor, ShareholderCountFactor,
)
from yinzi.factor_base import FactorCategory, FactorResult


def _make_daily_data(n_days=60, close_prices=None):
    """构造模拟日线数据。"""
    if close_prices is None:
        close_prices = [Decimal(str(10 + i * 0.1)) for i in range(n_days)]
    data = {}
    for i in range(n_days):
        date_str = f"2026{(i//30)+1:02d}{(i%30)+1:02d}"
        c = close_prices[i]
        data[date_str] = {
            "open": c - Decimal("0.1"),
            "high": c + Decimal("0.2"),
            "low": c - Decimal("0.2"),
            "close": c,
            "volume": Decimal("10000000"),
            "amount": Decimal("100000000"),
            "turnover_rate": Decimal("1.0"),
        }
    return data


class TestValueFactors:
    def test_pe_from_financial(self):
        f = PEFactor()
        result = f.compute("000001", {}, {"pe": 15.5})
        assert result == Decimal("15.5")

    def test_pe_missing(self):
        assert PEFactor().compute("000001", {}, {}) is None

    def test_pb(self):
        assert PBFactor().compute("000001", {}, {"pb": 2.1}) == Decimal("2.1")

    def test_fcf_yield(self):
        assert FCFYieldFactor().compute("000001", {}, {"free_cashflow_yield": 0.05}) == Decimal("0.05")

    def test_factor_metadata(self):
        f = PEFactor()
        assert f.name == "pe"
        assert f.category == FactorCategory.VALUE
        assert f.direction == -1


class TestMomentumFactors:
    def test_1m_momentum_positive(self):
        daily = _make_daily_data(30, [Decimal(str(10 + i * 0.15)) for i in range(30)])
        result = Momentum1MFactor().compute("000001", daily)
        assert result is not None
        assert result > 0

    def test_1m_momentum_negative(self):
        daily = _make_daily_data(30, [Decimal(str(20 - i * 0.1)) for i in range(30)])
        result = Momentum1MFactor().compute("000001", daily)
        assert result is not None
        assert result < 0

    def test_insufficient_data(self):
        daily = _make_daily_data(10)
        assert Momentum1MFactor().compute("000001", daily) is None

    def test_12m1m(self):
        daily = _make_daily_data(260, [Decimal(str(10 + i * 0.05)) for i in range(260)])
        result = Momentum12M1MFactor().compute("000001", daily)
        assert result is not None
        assert result > 0


class TestQualityFactors:
    def test_roe(self):
        assert ROEFactor().compute("000001", {}, {"roe": 15.2}) == Decimal("15.2")

    def test_cashflow_to_revenue(self):
        result = CashflowToRevenueFactor().compute(
            "000001", {}, {"operating_cashflow": 100, "revenue": 1000}
        )
        assert result == Decimal("0.1")


class TestVolatilityFactors:
    def test_historical_vol(self):
        daily = _make_daily_data(60)
        result = HistoricalVolFactor().compute("000001", daily)
        assert result is not None
        assert result >= 0

    def test_downside_vol(self):
        # 构造有涨有跌且跌幅各异的价格序列
        prices = []
        base = Decimal("10")
        for i in range(60):
            if i % 3 == 0:
                base -= Decimal(str(0.1 + (i % 5) * 0.05))  # 各异跌幅
            else:
                base += Decimal("0.15")
            prices.append(base)
        daily = _make_daily_data(60, prices)
        result = DownsideVolFactor().compute("000001", daily)
        assert result is not None

    def test_beta_needs_market(self):
        assert BetaFactor().compute("000001", _make_daily_data(60)) is None

    def test_beta_with_market(self):
        stock = _make_daily_data(100)
        market = _make_daily_data(100)
        result = BetaFactor().compute("000001", stock, market_data=market)
        assert result is not None

    def test_var(self):
        daily = _make_daily_data(100)
        result = VaRFactor().compute("000001", daily)
        assert result is not None
        assert result > 0


class TestSentimentFactors:
    def test_volume_ratio(self):
        daily = _make_daily_data(30)
        result = VolumeRatioFactor().compute("000001", daily)
        assert result is not None

    def test_turnover_anomaly(self):
        daily = _make_daily_data(30)
        result = TurnoverAnomalyFactor().compute("000001", daily)
        assert result is not None

    def test_north_bound_from_kwargs(self):
        result = NorthBoundFactor().compute("000001", {}, north_bound_data={"000001": 100})
        assert result == Decimal("100")


class TestAlternativeFactors:
    def test_analyst_coverage(self):
        result = AnalystCoverageFactor().compute(
            "000001", {}, analyst_coverage={"000001": (15, 12)}
        )
        assert result == Decimal("0.25")

    def test_shareholder_count_decrease(self):
        result = ShareholderCountFactor().compute(
            "000001", {}, shareholder_count={"000001": (80000, 100000)}
        )
        # 股东人数 100000→80000, 变化率 = (80000-100000)/100000 = -0.20
        assert result == Decimal("-0.20")

    def test_missing_kwargs(self):
        assert InstitutionalHoldingFactor().compute("000001", {}) is None


class TestFactorResult:
    def test_result_dataclass(self):
        r = FactorResult(code="000001", factor_name="pe", category=FactorCategory.VALUE, raw_value=Decimal("15"))
        assert r.code == "000001"
        assert r.z_score is None

    def test_batch_compute(self):
        f = PEFactor()
        fin_map = {"000001": {"pe": 15}, "000002": {"pe": 25}}
        results = f.compute_batch(["000001", "000002"], {}, fin_map)
        assert len(results) == 2
        assert results[0].raw_value == Decimal("15")
