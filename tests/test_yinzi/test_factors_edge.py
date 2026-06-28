"""
因子边缘场景测试：空数据、单股票、极端值、大数据集、FactorEngine。
"""

import time
from decimal import Decimal

from yinzi.engine import create_default_engine
from yinzi.factor_validator import FactorValidator
from yinzi.momentum_factors import Momentum1MFactor
from yinzi.value_factors import PEFactor, PEGFactor
from yinzi.volatility_factors import HistoricalVolFactor


class TestEdgeCases:
    def test_empty_daily_data(self):
        assert Momentum1MFactor().compute("000001", {}) is None
        assert HistoricalVolFactor().compute("000001", {}) is None

    def test_single_day_data(self):
        daily = {"20260601": {"open": "10", "close": "10.5", "volume": "1000", "amount": "10000"}}
        assert Momentum1MFactor().compute("000001", daily) is None
        assert HistoricalVolFactor().compute("000001", daily) is None

    def test_zero_price_handling(self):
        daily = {f"2026{i:02d}01": {"close": 0, "open": 0, "volume": 0, "amount": 0, "turnover_rate": 0} for i in range(1, 31)}
        result = Momentum1MFactor().compute("000001", daily)
        assert result is None  # division by zero → returns None

    def test_negative_financial_values(self):
        result = PEFactor().compute("000001", {}, {"pe": -5.0})
        assert result == Decimal("-5.0")

    def test_large_batch_performance(self):
        f = PEFactor()
        stocks = [f"{i:06d}" for i in range(1, 501)]
        fin_map = {c: {"pe": 10 + (i % 30)} for i, c in enumerate(stocks)}
        t0 = time.perf_counter()
        results = f.compute_batch(stocks, {}, fin_map)
        elapsed = time.perf_counter() - t0
        assert len(results) == 500
        assert elapsed < 2.0, f"Batch too slow: {elapsed:.2f}s"

    def test_validator_all_identical(self):
        n = 50
        fv = {f"{i:06d}": Decimal("1") for i in range(n)}
        fr = {f"{i:06d}": Decimal(str(i * 0.001)) for i in range(n)}
        ic = FactorValidator.compute_rank_ic(fv, fr)
        assert ic is not None

    def test_peg_with_historical_growth(self):
        factor = PEGFactor()
        historical = {
            "000001": [
                {"net_profit": 100, "report_date": "20250331"},
                {"net_profit": 125, "report_date": "20260331"},
            ]
        }
        result = factor.compute("000001", {}, {"pe": 20}, historical_financials=historical)
        assert result is not None
        assert result > 0

    def test_peg_no_growth_returns_none(self):
        factor = PEGFactor()
        result = factor.compute("000001", {}, {"pe": 20})
        assert result is None


class TestFactorEngine:
    def test_create_default(self):
        engine = create_default_engine(max_workers=2)
        assert engine.factor_count == 26

    def test_compute_all_sequential(self):
        engine = create_default_engine(max_workers=2)
        stocks = [f"{i:06d}" for i in range(1, 21)]
        fin_map = {c: {"pe": 10 + i, "roe": 5 + i * 0.5} for i, c in enumerate(stocks)}
        results = engine.compute_all(stocks, {}, fin_map, parallel=False)
        assert len(results) > 0  # PE + ROE 应有值

    def test_compute_by_category(self):
        engine = create_default_engine(max_workers=2)
        stocks = [f"{i:06d}" for i in range(1, 11)]
        fin_map = {c: {"pe": 10 + i} for i, c in enumerate(stocks)}
        results = engine.compute_category("value", stocks, {}, fin_map)
        assert len(results) > 0

    def test_compute_all_parallel(self):
        engine = create_default_engine(max_workers=4)
        stocks = [f"{i:06d}" for i in range(1, 31)]
        fin_map = {c: {"pe": 10 + i, "roe": 5 + i * 0.5} for i, c in enumerate(stocks)}
        results = engine.compute_all(stocks, {}, fin_map, parallel=True)
        assert len(results) > 0

    def test_vectorized_vs_batch_same_results(self):
        """向量化结果应与批量结果一致。"""
        f = PEFactor()
        stocks = [f"{i:06d}" for i in range(1, 21)]
        fin_map = {c: {"pe": 10 + i} for i, c in enumerate(stocks)}
        batch = f.compute_batch(stocks, {}, fin_map)
        vectorized = f.compute_vectorized(stocks, {}, fin_map)
        assert len(batch) == len(vectorized)
        assert {r.code: r.raw_value for r in batch} == {r.code: r.raw_value for r in vectorized}
