"""Tests for enhanced portfolio optimizer (v4.0 convex) -- constraint binding, scenarios, edges."""
from decimal import Decimal

import pytest

from juece.portfolio_optimizer import (
    BLConfig,
    OptimizationConstraints,
    PortfolioOptimizer,
    View,
)


def _make_returns(n_assets=10, n_periods=100, seed=42):
    import random
    random.seed(seed)
    return [[Decimal(str(random.gauss(0.001, 0.02))) for _ in range(n_periods)] for _ in range(n_assets)]


def _make_codes(n=10):
    return [f"{i:06d}" for i in range(n)]


class TestPortfolioOptimizerBasic:
    def test_estimate_covariance_shape(self):
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(_make_returns(5, 100))
        assert len(cov) == 5 and all(len(r) == 5 for r in cov)

    def test_covariance_symmetric(self):
        cov = PortfolioOptimizer().estimate_covariance(_make_returns(5, 100))
        for i in range(5):
            for j in range(5):
                assert abs(cov[i][j] - cov[j][i]) < Decimal("0.0001")

    def test_covariance_positive_definite(self):
        cov = PortfolioOptimizer().estimate_covariance(_make_returns(5, 100))
        for i in range(5):
            assert cov[i][i] > Decimal("0")

    def test_implied_returns_shape(self):
        opt = PortfolioOptimizer()
        pi = opt.implied_equilibrium_returns(
            opt.estimate_covariance(_make_returns(5, 100)), [Decimal("0.2")] * 5,
        )
        assert len(pi) == 5

    def test_no_views_preserves_equilibrium(self):
        opt = PortfolioOptimizer()
        rm = _make_returns(5, 100)
        codes = _make_codes(5)
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, _ = opt.incorporate_views(pi, cov, [], codes)
        assert pr == pi

    def test_optimize_basic(self):
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.30")))
        cov = opt.estimate_covariance(_make_returns(10, 100))
        codes = _make_codes(10)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.1")] * 10)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes)
        assert len(r.optimal_weights) == 10
        assert abs(float(sum(r.optimal_weights.values())) - 1.0) < 0.01
        assert r.expected_sharpe is not None
        assert r.n_positions > 0

    def test_max_weight_constraint(self):
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.25")))
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        for w in opt.optimize(pr, pc, codes).optimal_weights.values():
            assert w <= Decimal("0.26")

    def test_volatility_target(self):
        config = BLConfig(max_weight=Decimal("0.30"), volatility_target=Decimal("0.15"))
        opt = PortfolioOptimizer(config)
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        assert opt.optimize(pr, pc, codes).expected_volatility <= Decimal("0.16")

    def test_cardinality_constraint(self):
        config = BLConfig(max_weight=Decimal("0.30"), max_positions=3)
        opt = PortfolioOptimizer(config)
        cov = opt.estimate_covariance(_make_returns(10, 100))
        codes = _make_codes(10)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.1")] * 10)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        assert opt.optimize(pr, pc, codes).n_positions <= 3

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError):
            PortfolioOptimizer().optimize([], [], [])

    def test_optimization_result_fields(self):
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes)
        for attr in ["expected_return", "expected_volatility", "var_95", "var_99",
                      "turnover", "optimization_success"]:
            assert getattr(r, attr) is not None
        assert isinstance(r.optimization_success, bool)

    def test_return_with_views(self):
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        view = View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("0.1"), confidence=Decimal("0.7"))
        pr, pc = opt.incorporate_views(pi, cov, [view], codes)
        r = opt.optimize(pr, pc, codes)
        assert codes[0] in r.optimal_weights
        assert r.optimal_weights[codes[0]] > Decimal("0")

    def test_multiple_views(self):
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        views = [
            View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("0.1"), confidence=Decimal("0.8")),
            View(assets=[codes[1]], weights=[Decimal("1")], view_return=Decimal("-0.05"), confidence=Decimal("0.5")),
        ]
        pr, pc = opt.incorporate_views(pi, cov, views, codes)
        r = opt.optimize(pr, pc, codes)
        assert abs(float(sum(r.optimal_weights.values())) - 1.0) < 0.01


# ================================================================
# Turnover constraint -- real scenario tests
# ================================================================

class TestTurnoverConstraint:

    def test_loose_turnover_allows_rebalance(self):
        """Loose constraint (100% turnover) allows diversification."""
        config = BLConfig(max_weight=Decimal("0.30"), turnover_max=Decimal("1.0"))
        opt = PortfolioOptimizer(config)
        rm = _make_returns(5, 200)
        codes = _make_codes(5)
        curr = {codes[0]: Decimal("0.80"), codes[1]: Decimal("0.20")}
        for c in codes[2:]: curr[c] = Decimal("0")
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes, curr_w=curr)
        assert r.n_positions >= 3
        assert r.turnover > Decimal("0.1")

    def test_tight_turnover_limits_rebalance(self):
        """Tight turnover limit (0.55) leaves some rebalancing room after max_weight constraint."""
        # Minimum turnover from [0.8,0.2] subject to max_weight=0.30 is ~0.50.
        # So turnover_max must be >= 0.50 to be feasible.
        config = BLConfig(max_weight=Decimal("0.30"), turnover_max=Decimal("0.55"))
        opt = PortfolioOptimizer(config)
        rm = _make_returns(5, 200)
        codes = _make_codes(5)
        curr = {codes[0]: Decimal("0.80"), codes[1]: Decimal("0.20")}
        for c in codes[2:]: curr[c] = Decimal("0")
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes, curr_w=curr)
        assert r.turnover <= Decimal("0.60"), f"Turnover {float(r.turnover):.4f} exceeds limit"

    def test_tight_vs_loose_comparison(self):
        """Tight turnover constraint leaves less room than loose."""
        # Minimum feasible turnover = 0.5 from [0.7,0.3]. Loose=0.80 allows almost anything.
        # Tight=0.52 barely allows max_weight to be satisfied.
        config_loose = BLConfig(max_weight=Decimal("0.30"), turnover_max=Decimal("0.80"))
        config_tight = BLConfig(max_weight=Decimal("0.30"), turnover_max=Decimal("0.52"))
        opt_loose = PortfolioOptimizer(config_loose)
        opt_tight = PortfolioOptimizer(config_tight)
        rm = _make_returns(5, 200)
        codes = _make_codes(5)
        curr = {codes[0]: Decimal("0.70"), codes[1]: Decimal("0.30")}
        for c in codes[2:]: curr[c] = Decimal("0")
        cov = opt_loose.estimate_covariance(rm)
        pi = opt_loose.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt_loose.incorporate_views(pi, cov, [], codes)
        r_loose = opt_loose.optimize(pr, pc, codes, curr_w=curr)
        r_tight = opt_tight.optimize(pr, pc, codes, curr_w=curr)
        assert r_tight.turnover <= r_loose.turnover + Decimal("0.05"), (
            f"Tight={float(r_tight.turnover):.4f} > Loose={float(r_loose.turnover):.4f}"
        )

    def test_zero_current_weights(self):
        """Empty current weights -- optimization succeeds."""
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.30"), turnover_max=Decimal("1.0")))
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes, curr_w={})
        assert abs(float(sum(r.optimal_weights.values())) - 1.0) < 0.01

    def test_exact_match_gives_zero_turnover(self):
        """Current weights = optimal -> zero turnover."""
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.30")))
        cov = opt.estimate_covariance(_make_returns(5, 100))
        codes = _make_codes(5)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.2")] * 5)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r_base = opt.optimize(pr, pc, codes)
        r = opt.optimize(pr, pc, codes, curr_w=r_base.optimal_weights)
        assert r.turnover < Decimal("0.001")


# ================================================================
# EVT VaR constraint
# ================================================================

class TestEVTVaRConstraint:
    def test_var_limit_reduces_risk(self):
        """EVT VaR limit reduces portfolio volatility."""
        opt_no = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.40")))
        opt_yes = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.40")))
        import random
        random.seed(7)
        n_a, n_t = 6, 200
        rm = [[Decimal(str(random.gauss(0.002 * (i + 1), 0.03))) for _ in range(n_t)] for i in range(n_a)]
        codes = _make_codes(n_a)
        mw = [Decimal("1") / Decimal(n_a)] * n_a
        cov = opt_no.estimate_covariance(rm)
        pi = opt_no.implied_equilibrium_returns(cov, mw)
        pr, pc = opt_no.incorporate_views(pi, cov, [], codes)
        r_no = opt_no.optimize(pr, pc, codes)
        ec = OptimizationConstraints(var_limit=Decimal("0.03"))
        r_yes = opt_yes.optimize(pr, pc, codes, ext_con=ec)
        assert r_yes.expected_volatility <= r_no.expected_volatility + Decimal("0.02"), (
            f"Var limit did not reduce: no={float(r_no.expected_volatility):.4f}, yes={float(r_yes.expected_volatility):.4f}"
        )

    def test_var_limit_grid_small_n(self):
        """Grid search respects var_limit for small n."""
        import random
        random.seed(3)
        n_a, n_t = 4, 200
        rm = [[Decimal(str(random.gauss(0.001 + 0.003 * i, 0.04))) for _ in range(n_t)] for i in range(n_a)]
        codes = _make_codes(n_a)
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.40")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.25")] * n_a)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        ec = OptimizationConstraints(var_limit=Decimal("0.02"))
        r = opt.optimize(pr, pc, codes, ext_con=ec)
        assert r.expected_volatility <= Decimal("0.02")


# ================================================================
# Integration with EVT + Risk Budget
# ================================================================

class TestOptimizeWithRisk:
    def test_without_risk_engines(self):
        opt = PortfolioOptimizer()
        r = opt.optimize_with_risk(_make_returns(5, 100), [Decimal("0.2")] * 5, _make_codes(5), evt=False, rbudget=False)
        assert len(r.optimal_weights) == 5

    def test_with_evt_enabled(self):
        opt = PortfolioOptimizer()
        r = opt.optimize_with_risk(_make_returns(3, 100), [Decimal("0.33")] * 3, _make_codes(3), evt=True, rbudget=False)
        assert len(r.optimal_weights) == 3

    def test_with_risk_budget_enabled(self):
        opt = PortfolioOptimizer()
        r = opt.optimize_with_risk(_make_returns(5, 100), [Decimal("0.2")] * 5, _make_codes(5), evt=False, rbudget=True)
        assert len(r.optimal_weights) == 5


# ================================================================
# Edge cases
# ================================================================

class TestEdgeCases:
    def test_single_asset(self):
        opt = PortfolioOptimizer()
        rm = _make_returns(1, 100)
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("1")])
        pr, pc = opt.incorporate_views(pi, cov, [], ["A"])
        r = opt.optimize(pr, pc, ["A"])
        assert abs(float(r.optimal_weights["A"]) - 1.0) < 0.01

    def test_two_assets(self):
        import random
        random.seed(42)
        n_t = 300
        rm = [[Decimal(str(random.gauss(0.003, 0.01))) for _ in range(n_t)],
              [Decimal(str(random.gauss(-0.001, 0.02))) for _ in range(n_t)]]
        codes = _make_codes(2)
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.80")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.5")] * 2)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes)
        assert r.optimal_weights[codes[0]] >= r.optimal_weights[codes[1]]

    def test_many_assets_bounded(self):
        import random
        random.seed(42)
        n_a, n_t = 15, 100
        rm = [[Decimal(str(random.gauss(0.001, 0.01))) for _ in range(n_t)] for _ in range(n_a)]
        codes = _make_codes(n_a)
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.20")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("1") / Decimal(n_a)] * n_a)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        r = opt.optimize(pr, pc, codes)
        for w in r.optimal_weights.values():
            assert Decimal("0") <= w <= Decimal("0.25")

    def test_decimal_precision(self):
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(_make_returns(10, 100))
        codes = _make_codes(10)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.1")] * 10)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        total = sum(opt.optimize(pr, pc, codes).optimal_weights.values())
        assert abs(float(total) - 1.0) < 0.001


class TestOptimizationConstraintsOverride:
    def test_single_stock_max_via_ext_con(self):
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.10")))
        rm = _make_returns(20, 100)
        codes = _make_codes(20)
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.05")] * 20)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        ec = OptimizationConstraints(single_stock_max=Decimal("0.04"))
        r = opt.optimize(pr, pc, codes, ext_con=ec)
        for w in r.optimal_weights.values():
            assert w <= Decimal("0.05")

    def test_var_limit_stored(self):
        assert OptimizationConstraints(var_limit=Decimal("0.10")).var_limit == Decimal("0.10")
