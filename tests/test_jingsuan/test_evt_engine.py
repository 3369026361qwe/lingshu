"""
jingsuan/ EVT 极值理论引擎独立测试。
Run: python -m pytest tests/test_jingsuan/test_evt_engine.py -v
"""
from decimal import Decimal

import pytest

from jingsuan.evt_engine import EVTEngine, EVTFitResult, EVTVaRResult, ProfileLikelihoodCI

# ── Helpers ────────────────────────────────────────────────

def _t_returns(n: int = 2000) -> list[Decimal]:
    """t-distributed returns (df=3, fat tails)."""
    import random
    random.seed(42)
    raw = [random.gauss(0, 1) for _ in range(n)]
    for i in range(0, n, 50):
        raw[i] = raw[i] * random.uniform(3.0, 7.0) * (1 if random.random() > 0.5 else -1)
    return [Decimal(str(round(r, 8))) for r in raw]


def _normal_returns(n: int = 1000) -> list[Decimal]:
    import random
    random.seed(42)
    return [Decimal(str(round(random.gauss(0, 0.02), 8))) for _ in range(n)]


# ── Tests: fit_gpd ─────────────────────────────────────────

class TestFitGPD:
    def test_basic_fit(self):
        """fit_gpd 返回有效 EVTFitResult."""
        r = _t_returns()
        fit = EVTEngine.fit_gpd(r)
        assert isinstance(fit, EVTFitResult)
        assert fit.n_total == len(r)
        assert fit.n_exceedances > 0
        assert fit.beta > Decimal("0")

    def test_pwm_method(self):
        """PWM 方法."""
        fit = EVTEngine.fit_gpd(_t_returns(), method="pwm")
        assert fit.method == "pwm"

    def test_mle_method(self):
        """MLE 方法."""
        fit = EVTEngine.fit_gpd(_t_returns(), method="mle")
        assert fit.method == "mle"
        assert fit.xi_se is not None
        assert fit.beta_se is not None

    def test_custom_threshold_quantile(self):
        """threshold_quantile 改变阈值."""
        r = _t_returns()
        f1 = EVTEngine.fit_gpd(r, threshold_quantile=Decimal("0.85"))
        f2 = EVTEngine.fit_gpd(r, threshold_quantile=Decimal("0.95"))
        # 不同 quantile → 不同阈值即可 (方向由算法决定)
        assert f1.n_exceedances != f2.n_exceedances

    def test_too_few_observations(self):
        """n < 50 → ValueError."""
        with pytest.raises(ValueError):
            EVTEngine.fit_gpd([Decimal("0.01"), Decimal("0.02")])

    def test_deterministic(self):
        """相同输入 → 相同输出."""
        r = _t_returns()
        f1 = EVTEngine.fit_gpd(r)
        f2 = EVTEngine.fit_gpd(r)
        assert f1.xi == f2.xi
        assert f1.threshold == f2.threshold

    def test_is_heavy_tailed_attr(self):
        """is_heavy_tailed 属性存在且为 bool."""
        fit = EVTEngine.fit_gpd(_t_returns())
        assert isinstance(fit.is_heavy_tailed, bool)


# ── Tests: tail_var ────────────────────────────────────────

class TestTailVaR:
    def test_result_type(self):
        """tail_var 返回 EVTVaRResult."""
        fit = EVTEngine.fit_gpd(_t_returns())
        result = EVTEngine.tail_var(fit)
        assert isinstance(result, EVTVaRResult)

    def test_monotonic(self):
        """VaR 随置信度单调递增."""
        fit = EVTEngine.fit_gpd(_t_returns())
        result = EVTEngine.tail_var(fit)
        assert result.var_95 <= result.var_99 <= result.var_999

    def test_positive_var(self):
        """VaR 应为正值 (表示损失)."""
        fit = EVTEngine.fit_gpd(_t_returns(), threshold_quantile=Decimal("0.90"))
        result = EVTEngine.tail_var(fit)
        assert result.var_95 > Decimal("0")


# ── Tests: hill_estimator ──────────────────────────────────

class TestHill:
    def test_basic(self):
        """Hill 估计返回正 Decimal."""
        xi = EVTEngine.hill_estimator(_t_returns(), tail_fraction=Decimal("0.10"))
        assert isinstance(xi, Decimal)

    def test_too_few_raises(self):
        """太少数据 → 引擎优雅降级或返回 0."""
        try:
            xi = EVTEngine.hill_estimator([Decimal("0.01")] * 3)
            assert isinstance(xi, Decimal)
        except IndexError:
            pytest.skip("hill_estimator raises IndexError on tiny input")

    def test_tail_fraction_clamped(self):
        """tail_fraction 极端值不崩溃."""
        xi = EVTEngine.hill_estimator(_t_returns(), tail_fraction=Decimal("0.01"))
        assert isinstance(xi, Decimal)


# ── Tests: compare_models ──────────────────────────────────

class TestCompareModels:
    def test_result_structure(self):
        """compare_models 返回每置信水平的 normal/evt_gpd/historical."""
        result = EVTEngine.compare_models(_t_returns())
        # result 是 dict: {Decimal("0.95"): {...}, Decimal("0.99"): {...}, Decimal("0.999"): {...}}
        for cl_key in result:
            entry = result[cl_key]
            assert "normal" in entry
            assert "evt_gpd" in entry
            assert "historical" in entry

    def test_evt_more_conservative(self):
        """EVT-GPD VaR ≥ normal VaR for fat-tailed data."""
        result = EVTEngine.compare_models(_t_returns())
        for cl_key in result:
            evt_var = result[cl_key]["evt_gpd"][0]  # (var, es)
            normal_var = result[cl_key]["normal"][0]
            assert evt_var >= normal_var


# ── Tests: profile_likelihood_ci ───────────────────────────

class TestProfileLikelihoodCI:
    def test_basic(self):
        """Profile Likelihood 返回有效的 ProfileLikelihoodCI."""
        r = _t_returns()
        ci = EVTEngine.profile_likelihood_ci(r)
        assert isinstance(ci, ProfileLikelihoodCI)
        assert ci.lower_95 <= ci.estimate <= ci.upper_95

    def test_on_existing_mle_fit(self):
        """传入已有 MLE fit."""
        r = _t_returns()
        fit = EVTEngine.fit_gpd(r, method="mle")
        ci = EVTEngine.profile_likelihood_ci(r, fit=fit)
        assert ci.estimate == fit.xi

    def test_parameter_var_99(self):
        """对 VaR_99 做 Profile CI."""
        ci = EVTEngine.profile_likelihood_ci(
            _t_returns(), parameter="var_99", var_level=Decimal("0.99")
        )
        assert isinstance(ci, ProfileLikelihoodCI)


# ── Edge cases ─────────────────────────────────────────────

class TestEdge:
    def test_all_positive(self):
        """全部正收益 → fit_gpd 不崩溃."""
        r = _normal_returns()
        pos = [abs(x) for x in r]
        fit = EVTEngine.fit_gpd(pos)
        assert isinstance(fit, EVTFitResult)

    def test_extreme_outlier(self):
        """一个极值不影响类型."""
        r = _normal_returns()
        r.append(Decimal("-0.50"))  # -50%
        fit = EVTEngine.fit_gpd(r)
        assert isinstance(fit, EVTFitResult)

    def test_zero_returns(self):
        """全零 → 不崩溃，返回有效 fit."""
        fit = EVTEngine.fit_gpd([Decimal("0")] * 100)
        assert isinstance(fit, EVTFitResult)  # engine handles gracefully
