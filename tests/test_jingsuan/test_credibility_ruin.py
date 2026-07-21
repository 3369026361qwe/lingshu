"""
jingsuan/ Credibility + Ruin engine 独立测试。
Run: python -m pytest tests/test_jingsuan/test_credibility_ruin.py -v
"""
from decimal import Decimal

from jingsuan.credibility import CredibilityEngine, SourceTrackRecord
from jingsuan.ruin_engine import RuinConfig, RuinEngine


def _make_ic(n: int = 100) -> list[Decimal]:
    import random
    random.seed(42)
    return [Decimal(str(round(random.gauss(0.03, 0.05), 8))) for _ in range(n)]


def _returns(n=300):
    import random
    random.seed(42)
    return [Decimal(str(round(random.gauss(0, 0.02), 8))) for _ in range(n)]


class TestCredibilityEngine:
    def test_buhlmann_straub_basic(self):
        sources = [
            SourceTrackRecord(name="factor", ic_values=_make_ic(252)),
            SourceTrackRecord(name="gnn", ic_values=_make_ic(120)),
        ]
        result = CredibilityEngine.buhlmann_straub(sources)
        assert result is not None
        assert hasattr(result, 'source_weights')
        assert hasattr(result, 'credibility_factors')

    def test_two_minimum(self):
        """Bühlmann-Straub 需要 >= 2 源."""
        import pytest
        sources = [SourceTrackRecord(name="factor", ic_values=_make_ic(252))]
        with pytest.raises(ValueError):
            CredibilityEngine.buhlmann_straub(sources)

    def test_fuse_signals(self):
        sources = [
            SourceTrackRecord(name="f1", ic_values=_make_ic(200)),
            SourceTrackRecord(name="f2", ic_values=_make_ic(200)),
        ]
        fused = CredibilityEngine.fuse_signals(sources)
        assert isinstance(fused, list)
        assert len(fused) == 2


class TestRuinEngine:
    def test_estimate_ruin_probability_basic(self):
        r = _returns(200)
        prob = RuinEngine.estimate_ruin_probability(r, Decimal("1000000"))
        assert Decimal("0") <= prob <= Decimal("1")

    def test_lower_capital_higher_risk(self):
        r = _returns(200)
        p1 = RuinEngine.estimate_ruin_probability(r, Decimal("1000000"))
        p2 = RuinEngine.estimate_ruin_probability(r, Decimal("100000"))
        assert p2 >= p1

    def test_optimal_position_size(self):
        r = _returns(200)
        config = RuinConfig(acceptable_ruin_prob=Decimal("0.01"))
        result = RuinEngine.optimal_position_size(r, config)
        assert result is not None

    def test_dynamic_risk_budget(self):
        budget = RuinEngine.dynamic_risk_budget(Decimal("0.05"), Decimal("10.0"))
        assert Decimal("0.01") <= budget <= Decimal("10.0")
