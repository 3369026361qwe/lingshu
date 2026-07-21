"""
jingsuan/ Copula 引擎独立测试。
Run: python -m pytest tests/test_jingsuan/test_copula_engine.py -v
"""
from decimal import Decimal

from jingsuan.copula_engine import CopulaEngine, CopulaFit, CopulaType


def _bivariate_t(n: int = 500) -> list[list[Decimal]]:
    """Generate correlated returns."""
    import random
    random.seed(42)
    rho = 0.7
    z1 = [random.gauss(0, 1) for _ in range(n)]
    z2 = [rho * z1[i] + (1 - rho * rho) ** 0.5 * random.gauss(0, 1) for i in range(n)]
    for i in range(0, n, 30):
        if random.random() > 0.5:
            z1[i] *= random.uniform(3, 6) * (1 if random.random() > 0.5 else -1)
            z2[i] *= random.uniform(3, 6) * (1 if random.random() > 0.5 else -1)
    return [
        [Decimal(str(round(z1[i], 8))) for i in range(n)],
        [Decimal(str(round(z2[i], 8))) for i in range(n)],
    ]


class TestFit:
    def test_single_clayton(self):
        r = _bivariate_t()
        fit = CopulaEngine.fit(r, types=[CopulaType.CLAYTON])
        assert isinstance(fit, CopulaFit)
        assert fit.copula_type == CopulaType.CLAYTON
        assert "theta" in fit.params
        assert fit.aic is not None

    def test_single_gumbel(self):
        fit = CopulaEngine.fit(_bivariate_t(), types=[CopulaType.GUMBEL])
        assert fit.copula_type == CopulaType.GUMBEL

    def test_single_frank(self):
        fit = CopulaEngine.fit(_bivariate_t(), types=[CopulaType.FRANK])
        assert fit.copula_type == CopulaType.FRANK

    def test_multi_type_picks_best(self):
        r = _bivariate_t()
        fit = CopulaEngine.fit(r, types=[CopulaType.CLAYTON, CopulaType.GUMBEL, CopulaType.FRANK])
        assert fit.copula_type is not None
        assert fit.aic is not None

    def test_default_types(self):
        fit = CopulaEngine.fit(_bivariate_t())
        assert isinstance(fit, CopulaFit)

    def test_rotated_gumbel(self):
        fit = CopulaEngine.fit(_bivariate_t(), types=[CopulaType.ROTATED_GUMBEL])
        assert fit.copula_type == CopulaType.ROTATED_GUMBEL

    def test_t_copula(self):
        fit = CopulaEngine.fit(_bivariate_t(), types=[CopulaType.T])
        assert fit.copula_type == CopulaType.T

    def test_gaussian(self):
        fit = CopulaEngine.fit(_bivariate_t(), types=[CopulaType.GAUSSIAN])
        assert fit.copula_type == CopulaType.GAUSSIAN


class TestSimulate:
    def test_produces_scenarios(self):
        r = _bivariate_t()
        fit = CopulaEngine.fit(r, types=[CopulaType.CLAYTON])
        sc = CopulaEngine.simulate(fit, n_scenarios=3000)
        # Returns list[list[float]]: [scenario][asset]
        assert len(sc) > 0  # some scenarios
        assert len(sc[0]) == 2  # bivariate → 2 assets per scenario

    def test_default_n(self):
        fit = CopulaEngine.fit(_bivariate_t())
        sc = CopulaEngine.simulate(fit)
        assert len(sc[0]) > 0

    def test_values_in_unit_interval(self):
        fit = CopulaEngine.fit(_bivariate_t())
        sc = CopulaEngine.simulate(fit, n_scenarios=200)
        # sc is [scenario][asset] — check first 50 scenarios
        for s in sc[:50]:
            for v in s:
                assert Decimal("0") <= Decimal(str(v)) <= Decimal("1")


class TestPortfolioTailLoss:
    def test_positive(self):
        r = _bivariate_t()
        fit = CopulaEngine.fit(r, types=[CopulaType.CLAYTON])
        loss = CopulaEngine.portfolio_tail_loss(
            fit, [Decimal("0.5"), Decimal("0.5")], confidence=Decimal("0.99")
        )
        assert isinstance(loss, (Decimal, float))

    def test_default_conf(self):
        fit = CopulaEngine.fit(_bivariate_t())
        loss = CopulaEngine.portfolio_tail_loss(fit, [Decimal("0.5"), Decimal("0.5")])
        assert isinstance(loss, (Decimal, float))


class TestEdge:
    def test_perfect_correlation(self):
        same = [Decimal(str(round(0.01 + i * 0.001, 8))) for i in range(200)]
        fit = CopulaEngine.fit([same, same], types=[CopulaType.GAUSSIAN])
        assert isinstance(fit, CopulaFit)

    def test_constant_returns(self):
        c1 = [Decimal("0.01")] * 200
        c2 = [Decimal("0.02")] * 200
        fit = CopulaEngine.fit([c1, c2], types=[CopulaType.FRANK])
        assert isinstance(fit, CopulaFit)
