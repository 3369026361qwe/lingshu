"""
测试因子计算: 估值/动量/质量/波动率/情绪/另类。
"""

from decimal import Decimal

from yinzi.alternative_factors import (
    AnalystCoverageFactor,
    InstitutionalHoldingFactor,
    ShareholderCountFactor,
)
from yinzi.factor_base import FactorCategory, FactorResult
from yinzi.momentum_factors import (
    Momentum1MFactor,
    Momentum12M1MFactor,
)
from yinzi.quality_factors import (
    CashflowToRevenueFactor,
    ROEFactor,
)
from yinzi.sentiment_factors import (
    NorthBoundFactor,
    TurnoverAnomalyFactor,
    VolumeRatioFactor,
)
from yinzi.value_factors import FCFYieldFactor, PBFactor, PEFactor
from yinzi.volatility_factors import (
    BetaFactor,
    DownsideVolFactor,
    HistoricalVolFactor,
    VaRFactor,
)


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


# ── FDR + GARCH validation ──────────────────────────────────────────────────


class TestValidateAllFDR:
    """validate_all() 批量因子检验 + FDR 校正。"""

    @staticmethod
    def _make_factor(name, n_stocks=100, mean_ic=0.03, ic_std=0.05, n_periods=24):
        """生成模拟因子数据。"""
        import random
        random.seed(hash(name) % 2**31)
        factor_values = {f"{i:06d}": Decimal(str(random.uniform(-2, 2))) for i in range(1, n_stocks + 1)}
        forward_returns = {f"{i:06d}": Decimal(str(random.gauss(0.001, 0.02))) for i in range(1, n_stocks + 1)}
        ic_series = [
            Decimal(str(random.gauss(mean_ic, ic_std))) for _ in range(n_periods)
        ]
        return {"name": name, "factor_values": factor_values, "forward_returns": forward_returns, "ic_series": ic_series}

    def test_validate_all_basic(self):
        """validate_all 对多因子运行，返回 FDR 分类。"""
        from yinzi.factor_validator import FactorValidator
        factors = [
            self._make_factor("alpha", mean_ic=0.04, ic_std=0.03),
            self._make_factor("beta", mean_ic=0.01, ic_std=0.05),
            self._make_factor("gamma", mean_ic=0.06, ic_std=0.02),
        ]
        report = FactorValidator.validate_all(factors, run_garch=False)
        assert report["n_factors"] == 3
        assert "fdr_method" in report
        assert "results" in report
        assert len(report["results"]) == 3
        # 至少有一个因子被标记分类
        for r in report["results"]:
            assert "factor_name" in r
            assert r.get("ic_pvalue") is not None
            assert "fdr_classification" in r
            assert r["fdr_classification"] in ("显著", "需审查")

    def test_validate_all_single_factor(self):
        """单因子时不跑 BH，直接给 p-value 和分类。"""
        from yinzi.factor_validator import FactorValidator
        factors = [self._make_factor("solo", mean_ic=0.05, ic_std=0.02)]
        report = FactorValidator.validate_all(factors, run_garch=False)
        assert report["n_factors"] == 1
        assert report["n_tested"] == 1
        assert len(report["results"]) == 1
        r = report["results"][0]
        assert r["fdr_classification"] in ("显著", "需审查")

    def test_validate_all_no_ic_series(self):
        """无 IC 序列时跳过 FDR，保持基本结果。"""
        import random

        from yinzi.factor_validator import FactorValidator
        random.seed(42)
        fv = {f"{i:06d}": Decimal(str(random.uniform(-2, 2))) for i in range(50)}
        fr = {f"{i:06d}": Decimal(str(random.gauss(0.001, 0.02))) for i in range(50)}
        factors = [{"name": "no_ic", "factor_values": fv, "forward_returns": fr}]
        report = FactorValidator.validate_all(factors, run_garch=False)
        assert report["n_factors"] == 1
        assert "adjusted_pvalue" not in report["results"][0]  # 没有 p-value

    def test_validate_all_significant_vs_review(self):
        """强因子标显著，弱因子标需审查。"""
        from yinzi.factor_validator import FactorValidator
        factors = [
            self._make_factor("strong", mean_ic=0.08, ic_std=0.02),
            self._make_factor("weak", mean_ic=0.001, ic_std=0.05),
        ]
        report = FactorValidator.validate_all(factors, run_garch=False)
        strong = report["results"][0]
        weak = report["results"][1]
        # 强因子的 IC t-stat 应该更大
        assert strong.get("ic_t_stat", 0) > weak.get("ic_t_stat", 0)


class TestGARCHOnIC:
    """validate_all() 对 IC 序列拟合 GARCH。"""

    @staticmethod
    def _make_garch_ic(n=100):
        """生成有波动率聚类的模拟 IC 序列。"""
        import random
        random.seed(42)
        sigma2 = 0.0004
        ic = []
        for _ in range(n):
            eps = random.gauss(0, sigma2**0.5)
            sigma2 = 0.00001 + 0.15 * eps**2 + 0.80 * sigma2
            ic.append(Decimal(str(eps)))
        return ic

    def test_garch_on_ic_series(self):
        """IC 序列 >= 50 时拟合 GARCH。"""
        import random

        from yinzi.factor_validator import FactorValidator
        random.seed(42)
        ic = self._make_garch_ic(100)
        fv = {f"{i:06d}": Decimal(str(random.uniform(-2, 2))) for i in range(50)}
        fr = {f"{i:06d}": Decimal(str(random.gauss(0.001, 0.02))) for i in range(50)}
        factors = [{"name": "garch_test", "factor_values": fv, "forward_returns": fr, "ic_series": ic}]
        report = FactorValidator.validate_all(factors, run_garch=True)
        r = report["results"][0]
        assert report["n_garch_fitted"] == 1
        assert "garch_model" in r
        assert r["garch_model"] == "GARCH(1,1)"
        assert "garch_persistence" in r
        assert "garch_cond_vol_mean" in r
        assert "garch_stability" in r
        assert r["garch_stability"] in ("稳定", "中等", "不稳定")

    def test_garch_skip_short_series(self):
        """IC 序列 < 50 时跳过 GARCH。"""
        import random

        from yinzi.factor_validator import FactorValidator
        random.seed(42)
        ic_short = [Decimal(str(random.gauss(0, 0.02))) for _ in range(24)]
        fv = {f"{i:06d}": Decimal(str(random.uniform(-2, 2))) for i in range(50)}
        fr = {f"{i:06d}": Decimal(str(random.gauss(0.001, 0.02))) for i in range(50)}
        factors = [{"name": "short_ic", "factor_values": fv, "forward_returns": fr, "ic_series": ic_short}]
        report = FactorValidator.validate_all(factors, run_garch=True)
        assert report["n_garch_fitted"] == 0

    def test_garch_disabled(self):
        """run_garch=False 时跳过。"""
        from yinzi.factor_validator import FactorValidator
        ic = self._make_garch_ic(100)
        import random
        random.seed(42)
        fv = {f"{i:06d}": Decimal(str(random.uniform(-2, 2))) for i in range(50)}
        fr = {f"{i:06d}": Decimal(str(random.gauss(0.001, 0.02))) for i in range(50)}
        factors = [{"name": "no_garch", "factor_values": fv, "forward_returns": fr, "ic_series": ic}]
        report = FactorValidator.validate_all(factors, run_garch=False)
        assert report["n_garch_fitted"] == 0
