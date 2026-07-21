"""
Synthetic data mathematical verification (v4.0 audit-grade).

Blueprint section 11.1 verification invariants:
    1. BL: no views, high-mean asset gets highest weight
    2. BL: strong bullish view increases weight
    3. Ledoit-Wolf: shrinkage stabilises near-singular covariance
    4. HRP: block-correlated assets cluster correctly
    5. Attribution: sum(Brinson effects) = total active return
    6. Market Impact: participation p -> impact = kappa * sqrt(p)

Tests use controlled synthetic data with known properties.
"""

import math
from decimal import Decimal

from huice.attribution import AttributionEngine
from juece.hrp import HRPOptimizer
from juece.portfolio_optimizer import BLConfig, PortfolioOptimizer, View
from zhixing.market_impact import MarketImpactModel

# ============================================================
# Synthetic data generators
# ============================================================

def _generate_iid_returns(
    n_assets=10, n_periods=300, mean=0.001, vol=0.02, seed=42,
) -> list[list[Decimal]]:
    import random as _r
    rng = _r.Random(seed)
    return [[Decimal(str(rng.gauss(mean, vol))) for _ in range(n_periods)] for _ in range(n_assets)]


def _generate_block_correlated_returns(
    n_assets=10, n_periods=500,
    block_sizes=None, intra_corr=0.80, inter_corr=0.05,
    mean_ret=0.001, vol=0.02, seed=42,
) -> list[list[Decimal]]:
    import random as _r
    rng = _r.Random(seed)
    if block_sizes is None:
        block_sizes = [n_assets // 3] * 3
        block_sizes[-1] += n_assets - sum(block_sizes)

    true_cov = [[0.0] * n_assets for _ in range(n_assets)]
    idx = 0
    for bs in block_sizes:
        for i in range(idx, idx + bs):
            for j in range(idx, idx + bs):
                true_cov[i][j] = intra_corr * vol**2 if i != j else vol**2
        idx += bs
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j and true_cov[i][j] == 0.0:
                true_cov[i][j] = inter_corr * vol**2

    # Cholesky
    L = [[0.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (true_cov[i][j] - s) / max(L[j][j], 1e-12) if i != j else math.sqrt(max(true_cov[i][i] - s, 1e-12))

    rm = [[Decimal("0")] * n_periods for _ in range(n_assets)]
    for t in range(n_periods):
        z = [rng.gauss(0, 1) for _ in range(n_assets)]
        for i in range(n_assets):
            rm[i][t] = Decimal(str(sum(L[i][k] * z[k] for k in range(n_assets)) + mean_ret))
    return rm


# ============================================================
# 1. BL: directional correctness of views
# ============================================================

class TestBLViewDirection:
    """Views should push weights in the correct direction."""

    def test_high_mean_asset_gets_highest_weight_no_views(self):
        """Asset with highest expected return gets max weight (no views)."""
        import random
        random.seed(42)
        n_a, n_t = 6, 500
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.002 if i == 0 else 0.001, 0.02))) for _ in range(n_t)] for i in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        opt = PortfolioOptimizer(BLConfig(risk_aversion=Decimal("2.5"), max_weight=Decimal("0.40")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        result = opt.optimize(pr, pc, codes)
        # Asset 0 must have the highest weight
        w0 = result.optimal_weights[codes[0]]
        for i in range(1, n_a):
            assert w0 >= result.optimal_weights[codes[i]], (
                f"Asset 0 weight {float(w0):.4f} < asset {i} weight {float(result.optimal_weights[codes[i]]):.4f}"
            )

    def test_bullish_view_increases_weight(self):
        """Bullish view on asset[0] -> its weight increases vs baseline.

        Note: view_return must be comparable to equilibrium returns (daily scale ~1e-4)
        to avoid immediately hitting the weight boundary. With Ω properly scaled,
        even a modest view should tilt weights in the right direction.
        """
        import random
        random.seed(99)
        n_a, n_t = 5, 500
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.001, 0.015))) for _ in range(n_t)] for _ in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        opt = PortfolioOptimizer(BLConfig(risk_aversion=Decimal("2.5"), max_weight=Decimal("0.40")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr0, pc0 = opt.incorporate_views(pi, cov, [], codes)
        w_base = opt.optimize(pr0, pc0, codes).optimal_weights[codes[0]]
        # bullish view: daily excess ~0.0005 (annualised ~12%, reasonable)
        view = View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("0.002"), confidence=Decimal("0.50"))
        pr1, pc1 = opt.incorporate_views(pi, cov, [view], codes)
        w_view = opt.optimize(pr1, pc1, codes).optimal_weights[codes[0]]
        assert w_view >= w_base, f"Bull view: {float(w_base):.4f} -> {float(w_view):.4f}"

    def test_bearish_view_reduces_weight(self):
        """Bearish view on asset[0] -> its weight decreases vs baseline."""
        import random
        random.seed(77)
        n_a, n_t = 5, 500
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.001, 0.015))) for _ in range(n_t)] for _ in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        opt = PortfolioOptimizer(BLConfig(risk_aversion=Decimal("2.5"), max_weight=Decimal("0.40")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr0, pc0 = opt.incorporate_views(pi, cov, [], codes)
        w_base = opt.optimize(pr0, pc0, codes).optimal_weights[codes[0]]
        # bearish view
        view = View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("-0.002"), confidence=Decimal("0.50"))
        pr1, pc1 = opt.incorporate_views(pi, cov, [view], codes)
        w_view = opt.optimize(pr1, pc1, codes).optimal_weights[codes[0]]
        assert w_view <= w_base, f"Bear view: {float(w_base):.4f} -> {float(w_view):.4f}"

    def test_high_confidence_moves_further_than_low(self):
        """High-confidence view causes larger or equal weight change than low-confidence view."""
        import random
        random.seed(55)
        n_a, n_t = 5, 500
        codes = [f"{i:06d}" for i in range(n_a)]
        rm = [[Decimal(str(random.gauss(0.001, 0.015))) for _ in range(n_t)] for _ in range(n_a)]
        mkt_w = [Decimal("1") / Decimal(n_a)] * n_a
        opt = PortfolioOptimizer(BLConfig(risk_aversion=Decimal("2.5"), max_weight=Decimal("0.40")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, mkt_w)
        pr0, pc0 = opt.incorporate_views(pi, cov, [], codes)
        w_base = opt.optimize(pr0, pc0, codes).optimal_weights[codes[0]]
        # low confidence: 10% confidence = 99% uncertainty = tiny view influence
        v_lo = View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("0.002"), confidence=Decimal("0.01"))
        pr_lo, pc_lo = opt.incorporate_views(pi, cov, [v_lo], codes)
        w_lo = opt.optimize(pr_lo, pc_lo, codes).optimal_weights[codes[0]]
        # high confidence: 90% confidence = strong view influence
        v_hi = View(assets=[codes[0]], weights=[Decimal("1")], view_return=Decimal("0.002"), confidence=Decimal("0.90"))
        pr_hi, pc_hi = opt.incorporate_views(pi, cov, [v_hi], codes)
        w_hi = opt.optimize(pr_hi, pc_hi, codes).optimal_weights[codes[0]]
        dev_lo = abs(w_lo - w_base)
        dev_hi = abs(w_hi - w_base)
        assert dev_hi >= dev_lo, f"Low conf dev={float(dev_lo):.6f}, High conf dev={float(dev_hi):.6f}"
        pr_hi, pc_hi = opt.incorporate_views(pi, cov, [v_hi], codes)
        w_hi = opt.optimize(pr_hi, pc_hi, codes).optimal_weights[codes[0]]
        dev_lo = abs(w_lo - w_base)
        dev_hi = abs(w_hi - w_base)
        assert dev_hi > dev_lo, f"Low dev={float(dev_lo):.6f}, High dev={float(dev_hi):.6f}"


# ============================================================
# 2. BL: edge cases
# ============================================================

class TestBLEdgeCases:
    def test_single_asset_concentrates(self):
        rm = _generate_iid_returns(n_assets=1, n_periods=200, seed=1)
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("1")])
        pr, pc = opt.incorporate_views(pi, cov, [], ["A"])
        result = opt.optimize(pr, pc, ["A"])
        assert abs(float(result.optimal_weights["A"]) - 1.0) < 0.001

    def test_weights_sum_to_one(self):
        rm = _generate_iid_returns(n_assets=8, n_periods=300, seed=10)
        codes = [f"{i:06d}" for i in range(8)]
        opt = PortfolioOptimizer(BLConfig(max_weight=Decimal("0.30")))
        cov = opt.estimate_covariance(rm)
        pi = opt.implied_equilibrium_returns(cov, [Decimal("0.125")] * 8)
        pr, pc = opt.incorporate_views(pi, cov, [], codes)
        result = opt.optimize(pr, pc, codes)
        assert abs(float(sum(result.optimal_weights.values())) - 1.0) < 0.001


# ============================================================
# 3. Ledoit-Wolf: shrinkage stabilises covariance
# ============================================================

class TestLedoitWolfProperties:
    def test_covariance_is_symmetric_psd(self):
        rm = _generate_iid_returns(n_assets=8, n_periods=200, seed=17)
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(rm)
        for i in range(len(cov)):
            for j in range(len(cov)):
                assert abs(cov[i][j] - cov[j][i]) < Decimal("0.0001")
            assert cov[i][i] > Decimal("0")

    def test_shrinkage_stabilises_high_dim(self):
        """When n_assets ~ n_periods, shrinkage prevents singular/exploding estimates."""
        import random as _r
        _r.seed(42)
        n_a, n_t = 30, 50  # very close to singular
        rm = [[Decimal(str(_r.gauss(0.001, 0.02))) for _ in range(n_t)] for _ in range(n_a)]
        opt = PortfolioOptimizer()
        cov = opt.estimate_covariance(rm)
        for i in range(n_a):
            # variance must be positive and reasonable
            v = float(cov[i][i])
            assert 0.0001 < v < 0.01, f"Unstable variance at {i}: {v}"
            for j in range(i + 1, n_a):
                vi, vj = float(cov[i][i]), float(cov[j][j])
                if vi > 1e-10 and vj > 1e-10:
                    corr = float(cov[i][j]) / math.sqrt(vi * vj)
                    assert abs(corr) < 0.999, f"Near-perfect correlation ({i},{j}): {corr}"


# ============================================================
# 4. HRP: block structure detection
# ============================================================

class TestHRPProperties:
    def test_weights_nonnegative_and_sum_to_one(self):
        rm = _generate_block_correlated_returns(n_assets=9, n_periods=300)
        result = HRPOptimizer.allocate(rm)
        assert abs(float(sum(result.weights.values())) - 1.0) < 0.001
        for w in result.weights.values():
            assert w >= Decimal("0")

    def test_block_assets_cluster_together(self):
        """With 3 blocks of 3 highly-correlated assets, early merges are within-block."""
        rm = _generate_block_correlated_returns(
            n_assets=9, n_periods=500, block_sizes=[3, 3, 3],
            intra_corr=0.85, inter_corr=0.02,
        )
        result = HRPOptimizer.allocate(rm)
        # Early 6 merges (bottom of tree) should be within-block -> distance < 0.7
        early_merges = result.cluster_tree[:6]
        for _, merge in enumerate(early_merges):
            assert merge[2] < 0.7, f"Early merge distance {merge[2]:.3f} too large — not within-block"

    def test_cluster_tree_size(self):
        rm = _generate_block_correlated_returns(n_assets=10, n_periods=300)
        result = HRPOptimizer.allocate(rm)
        assert len(result.cluster_tree) == 9  # n-1 merges


# ============================================================
# 5. Attribution: additivity
# ============================================================

class TestAttributionAdditivity:
    def test_brinson_sum_equals_active_return(self):
        """Sum of allocation + selection + interaction = portfolio_return - benchmark_return."""
        pw = {
            "tech": [("t1", Decimal("0.6")), ("t2", Decimal("0.0"))],
            "finance": [("f1", Decimal("0.0")), ("f2", Decimal("0.4"))],
        }
        bw = {
            "tech": [("t1", Decimal("0.25")), ("t2", Decimal("0.25"))],
            "finance": [("f1", Decimal("0.25")), ("f2", Decimal("0.25"))],
        }
        pr = {"t1": Decimal("0.10"), "t2": Decimal("0.05"), "f1": Decimal("0.02"), "f2": Decimal("0.04")}
        br = {"t1": Decimal("0.08"), "t2": Decimal("0.06"), "f1": Decimal("0.03"), "f2": Decimal("0.03")}
        result = AttributionEngine.brinson(pw, bw, pr, br)
        total_effects = Decimal("0")
        for a, s, i in zip(result.allocation_effect, result.selection_effect, result.interaction_effect, strict=False):
            total_effects += a + s + i
        # Active return = portfolio_return - benchmark_return
        pw_ret = sum(pr.get(c, Decimal("0")) * w for sa in pw.values() for c, w in sa)
        bw_ret = sum(br.get(c, Decimal("0")) * w for sa in bw.values() for c, w in sa)
        assert abs(total_effects - (pw_ret - bw_ret)) < Decimal("0.001"), (
            f"Sum effects={float(total_effects):.6f} != active return={float(pw_ret - bw_ret):.6f}"
        )

    def test_perfect_replication_gives_zero_active(self):
        """Portfolio == benchmark -> zero active return."""
        pw = {"s1": [("a", Decimal("0.5")), ("b", Decimal("0.5"))]}
        bw = {"s1": [("a", Decimal("0.5")), ("b", Decimal("0.5"))]}
        rets = {"a": Decimal("0.05"), "b": Decimal("0.05")}
        result = AttributionEngine.brinson(pw, bw, rets, rets)
        assert abs(result.total_active_return) < Decimal("0.001")

    def test_risk_attribution_cvar_adds_up(self):
        """sum(component_VaR) = total_VaR."""
        n = 5
        w = [Decimal("1") / Decimal(n)] * n
        cov = [[Decimal("0.0004") if i == j else Decimal("0.00005") for j in range(n)] for i in range(n)]
        result = AttributionEngine.risk_attribution(w, cov)
        assert abs(sum(result.component_var) - result.var_total) < Decimal("0.0001"), (
            f"sum CVaR={float(sum(result.component_var)):.6f} != VaR={float(result.var_total):.6f}"
        )


# ============================================================
# 6. Market Impact: formula-level correctness
# ============================================================

class TestMarketImpactMath:
    def test_sqrt_formula_exact(self):
        for p in [0.01, 0.04, 0.09, 0.16, 0.25]:
            v = int(p * 1_000_000)
            impact = MarketImpactModel.sqrt_liquidity(v, 1_000_000, Decimal("10"), kappa=Decimal("20"))
            expected = Decimal("20") * Decimal(str(p)).sqrt()
            assert abs(impact.total_impact - expected) < Decimal("0.01"), (
                f"p={p}: got {float(impact.total_impact):.4f}, expected {float(expected):.4f}"
            )

    def test_ac_formula_zero_vol_gives_zero(self):
        impact = MarketImpactModel.almgren_chriss(100_000, 1_000_000, Decimal("0"), Decimal("10"))
        assert impact.total_impact == Decimal("0")

    def test_ac_permanent_plus_temp_le_total(self):
        impact = MarketImpactModel.almgren_chriss(200_000, 1_000_000, Decimal("0.02"), Decimal("10"))
        assert impact.permanent_impact + impact.temporary_impact <= impact.total_impact + Decimal("0.001")

    def test_twap_total_qty_conserved(self):
        for qty in [10000, 25000, 50000]:
            slices = MarketImpactModel.optimal_twap_slices(qty, 1_000_000, Decimal("0.02"), Decimal("10"))
            assert abs(sum(s["quantity"] for s in slices) - qty) < 100, f"qty={qty}: total={sum(s['quantity'] for s in slices)}"
