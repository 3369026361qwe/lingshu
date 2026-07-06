"""
Synthetic data verification for jingsuan v4.1 engines.
Run: PYTHONPATH=. python tests/verify_jingsuan_v41.py
"""
import math
import random
from decimal import Decimal

random.seed(42)
passed = 0
failed = 0

def check(name, condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}: {msg}")

print("=" * 60)
print("SYNTHETIC DATA VERIFICATION -- jingsuan v4.1")
print("=" * 60)

# -- 1. EVT Engine --
print("\n[1/7] EVT Engine")
from jingsuan.evt_engine import EVTEngine

xi_true = 0.35
beta_true = 1.2
n_total = 2000
n_exc = 300

losses_gpd = []
for _ in range(n_exc):
    u = random.random()
    loss_val = (beta_true / xi_true) * (u ** (-xi_true) - 1)
    losses_gpd.append(loss_val)

all_returns = []
for loss in losses_gpd:
    all_returns.append(Decimal(str(round(-loss, 8))))
for _ in range(n_total - n_exc):
    all_returns.append(Decimal(str(round(random.gauss(0, 0.01), 8))))

fit_pwm = EVTEngine.fit_gpd(all_returns, threshold_quantile=Decimal("0.70"), method="pwm")
check("PWM fit returns EVTFitResult", fit_pwm is not None)
check(f"PWM beta={round(float(fit_pwm.beta), 3)} > 0", float(fit_pwm.beta) > 0)

fit_mle = EVTEngine.fit_gpd(all_returns, threshold_quantile=Decimal("0.70"), method="mle")
check("MLE fit returns EVTFitResult", fit_mle is not None)

var_result = EVTEngine.tail_var(fit_pwm)
check(f"VaR_99={round(float(var_result.var_99), 4)} > 0", float(var_result.var_99) > 0)

ci = EVTEngine.profile_likelihood_ci(all_returns, fit_mle, parameter="xi")
check("Profile CI computed", float(ci.lower_95) <= float(ci.upper_95))

opt_thresh, diag = EVTEngine.auto_threshold(all_returns)
check("Auto threshold OK", float(opt_thresh) > 0)

cmp = EVTEngine.compare_models(all_returns)
check("Model compare has 3 levels", "0.95" in cmp and "0.99" in cmp)

n_ret_gev = 2000
gev_returns = [Decimal(str(round(random.gauss(-0.0005, 0.02), 6))) for _ in range(n_ret_gev)]
try:
    fit_gev = EVTEngine.fit_gev(gev_returns)
    check(f"GEV from EVTEngine: mu={round(float(fit_gev.mu), 3)}, tail={fit_gev.tail_type}", fit_gev is not None)
except Exception as e:
    check("GEV from EVTEngine", False, str(e))

# -- 2. GEV Engine --
print("\n[2/7] GEV Engine")
from jingsuan.gev_engine import GEVEngine

n_blocks = 100
xi_gev = 0.2
sigma_gev = 0.5
mu_gev = 2.0
block_maxima = []
for _ in range(n_blocks):
    u = random.random()
    z = mu_gev - (sigma_gev / xi_gev) * (1 - (-math.log(u)) ** (-xi_gev))
    block_maxima.append(Decimal(str(round(z, 8))))

fit_gev = GEVEngine.fit(block_maxima, method="pwm")
check("GEV PWM fit", fit_gev is not None)
check("Tail type returned", len(fit_gev.tail_type) > 0)

rl = GEVEngine.return_level(fit_gev, period=100)
check(f"Return Level 100={round(float(rl.return_level), 3)} > 0", float(rl.return_level) > 0)

qq = GEVEngine.quantile_plot(block_maxima, fit_gev)
check(f"Q-Q plot has {len(qq)} points", len(qq) == n_blocks)

pp = GEVEngine.probability_plot(block_maxima, fit_gev)
check(f"P-P plot has {len(pp)} points", len(pp) > 0)

compare_ge = GEVEngine.compare_gev_gpd(gev_returns)
check("GEV vs GPD comparison", "gpd" in compare_ge)

# -- 3. Copula Engine --
print("\n[3/7] Copula Engine")
from jingsuan.copula_engine import CopulaEngine, CopulaType

n_cop = 800
theta_cl = 2.0
u_pairs = []
for _ in range(n_cop):
    u1 = random.random()
    u2 = random.random()
    v = (u1 ** (-theta_cl / (1 + theta_cl)) * (u2 ** (-theta_cl) - 1) + 1) ** (-1 / theta_cl)
    u_pairs.append((Decimal(str(u1)), Decimal(str(v))))
rmat_cl = [[r1 for r1, _ in u_pairs], [r2 for _, r2 in u_pairs]]

fit_cop = CopulaEngine.fit(rmat_cl)
check(f"Best copula: {fit_cop.copula_type.value}", fit_cop is not None)
check("CvM p-value computed", fit_cop.cvm_pvalue >= 0)

fit_rg = CopulaEngine.fit(rmat_cl, types=[CopulaType.ROTATED_GUMBEL])
check(f"Rotated Gumbel LT dep={round(float(fit_rg.lower_tail_dep), 3)}", fit_rg.copula_type == CopulaType.ROTATED_GUMBEL)

cmp_all = CopulaEngine.compare_all(rmat_cl)
check(f"Compare all: {len(cmp_all)} copulas", len(cmp_all) >= 4)

rmat_3d = [[Decimal(str(random.gauss(0, 0.02))) for _ in range(400)] for _ in range(3)]
fit_multi = CopulaEngine.fit(rmat_3d, multi_dimensional=True)
check(f"Multi-dim: d={fit_multi.n_dimensions}", fit_multi.n_dimensions == 3)

td_mat = CopulaEngine.tail_dependence_matrix(rmat_cl)
check("Tail dep matrix", len(td_mat) == 2)

pl = CopulaEngine.portfolio_tail_loss(fit_cop, [Decimal("0.6"), Decimal("0.4")], n_scenarios=500)
check("Portfolio tail loss computed", pl is not None)

cond = CopulaEngine.conditional_sampling(fit_cop, {0: 0.05}, n_scenarios=500)
check(f"Conditional sampling: {len(cond)} scenarios", len(cond) == 500)

# -- 4. DCC Copula --
print("\n[4/7] DCC Copula Engine")
from jingsuan.dcc_copula import DCCCopula

n_dcc = 200
dcc_data = []
for _a in range(2):
    vals = []
    r_val = 0.0
    for _ in range(n_dcc):
        r_val = 0.01 * r_val + random.gauss(0.0001, 0.015)
        vals.append(Decimal(str(round(r_val, 6))))
    dcc_data.append(vals)

fit_dcc = DCCCopula.fit(dcc_data)
check("DCC fit successful", fit_dcc is not None)
check(f"DCC corr series length={len(fit_dcc.correlation_series(0, 1))}", len(fit_dcc.correlation_series(0, 1)) == n_dcc)

td_series = fit_dcc.tail_dependence_series(0, 1)
check(f"Tail dep series: {len(td_series)} time points", len(td_series) == n_dcc)

sim_dcc = DCCCopula.simulate(fit_dcc, n_scenarios=500)
check(f"DCC simulate: {len(sim_dcc)} scenarios", len(sim_dcc) == 500)

dynamic_td = DCCCopula.dynamic_tail_dependence(fit_dcc)
check(f"Dynamic TD: {len(dynamic_td)} windows", len(dynamic_td) == n_dcc)

# -- 5. Ruin Engine --
print("\n[5/7] Ruin Engine")
from jingsuan.ruin_engine import RuinConfig, RuinEngine

n_trades = 400
trade_returns = [Decimal(str(round(random.gauss(0.001, 0.02), 6))) for _ in range(n_trades)]
config = RuinConfig(
    initial_capital=Decimal("100000"), ruin_threshold=Decimal("50000"),
    n_simulations=2000, time_horizon=100,
)

psi = RuinEngine.estimate_ruin_probability(trade_returns, Decimal("0.5"), config)
check("Ruin prob computed", 0 <= float(psi) <= 1)

psi_cl = RuinEngine.cramer_lundberg_exact(trade_returns, Decimal("0.3"), config)
check("CL exact computed", 0 <= float(psi_cl) <= 1)

psi_bb = RuinEngine.beekman_bowers_approx(trade_returns, Decimal("0.3"), config)
check("Beekman-Bowers computed", 0 <= float(psi_bb) <= 1)

mp = RuinEngine.multi_period_ruin(trade_returns, Decimal("0.3"), config)
check(f"Multi-period: {len(mp.periods)} periods", len(mp.periods) >= 2)
check("All ruin_probs positive", all(float(p) >= 0 for p in mp.ruin_probs))

opt = RuinEngine.optimal_position_size(trade_returns, config)
check(f"Optimal f={round(float(opt.optimal_position_size), 4)} > 0", float(opt.optimal_position_size) > 0)

updated = RuinEngine.update_ruin_belief(mp, survived_days=21)
check("Bayesian update returns MultiPeriodRuin", updated is not None)

# -- 6. Credibility Engine --
print("\n[6/7] Credibility Engine")
from jingsuan.credibility import CredibilityEngine, SourceTrackRecord

sources = [
    SourceTrackRecord(name="yinzi", ic_values=[Decimal(str(round(random.gauss(0.05, 0.08), 6))) for _ in range(100)]),
    SourceTrackRecord(name="gnn", ic_values=[Decimal(str(round(random.gauss(0.02, 0.15), 6))) for _ in range(100)]),
    SourceTrackRecord(name="agent", ic_values=[Decimal(str(round(random.gauss(0.06, 0.10), 6))) for _ in range(80)]),
]

weights_bs = CredibilityEngine.buhlmann_straub(sources)
total_w = sum(float(w) for w in weights_bs.source_weights.values())
check(f"B-S weights sum={total_w:.4f}", abs(total_w - 1.0) < 0.001)

weights_td = CredibilityEngine.decay_weighted_credibility(sources, half_life=14)
check("Decay-weighted Z factors computed", "yinzi" in weights_td.credibility_factors)

regime_labels = [0 if random.random() < 0.6 else 1 for _ in range(100)]
hier = CredibilityEngine.hierarchical_credibility(sources, regime_labels=regime_labels)
check(f"Hierarchical: level={hier.level}", hier.level == "hierarchical")

hach = CredibilityEngine.hachemeister_regression(sources)
check(f"Hachemeister: {len(hach.blended_predictions)} sources blended", len(hach.blended_predictions) >= 2)

updated_src = CredibilityEngine.update_track_record(sources[0], Decimal("0.03"), Decimal("100"), "2026-07-07")
check(f"Track record updated: {updated_src.n_periods}", updated_src.n_periods == sources[0].n_periods + 1)

fused = CredibilityEngine.fuse_signals(sources)
check(f"Fuse signals: {len(fused)} weights", len(fused) == 3)

# -- 7. Reserving Engine --
print("\n[7/8] Reserving Engine")
from jingsuan.reserving import LossTriangle, ReservingEngine

data_fixed = []
for row in [
    [100000, 150000, 180000, 190000, 195000],
    [110000, 160000, 185000, 195000, None],
    [105000, 155000, 182000, None, None],
    [120000, 170000, None, None, None],
    [115000, None, None, None, None],
]:
    data_fixed.append([Decimal(str(v)) if v is not None else None for v in row])
triangle = LossTriangle.from_matrix(data_fixed)
cl_result = ReservingEngine.chain_ladder(triangle)
check("CL total_reserve > 0", float(cl_result.total_reserve) > 0)

bf_result = ReservingEngine.bornhuetter_ferguson(
    triangle, prior_ultimate=[Decimal("200000")] * 5, credibility=Decimal("0.5")
)
check("BF total_reserve > 0", float(bf_result.total_reserve) > 0)

patterns = ReservingEngine.development_pattern(triangle)
check(f"Dev pattern: {len(patterns)} years", len(patterns) >= 1)

try:
    reserve_range = ReservingEngine.reserve_range(triangle, n_bootstrap=100)
    check("Reserve range: mean present", "mean" in reserve_range)
except Exception as e:
    check("Reserve range", False, str(e))

premiums = [Decimal("200000")] * 5
cc_result = ReservingEngine.cape_cod(triangle, premiums)
check("Cape Cod executed", cc_result is not None)

# -- 8. VaR Backtest --
print("\n[8/8] VaR Backtest")
from jingsuan.var_backtest import VaRBacktestSuite

n_var = 500
var_forecasts = []
actual_losses = []
for _ in range(n_var):
    var_v = Decimal(str(round(max(random.gauss(0.03, 0.005), 0.01), 6)))
    var_forecasts.append(var_v)
    if random.random() < 0.01:
        loss = var_v + Decimal(str(random.uniform(0.001, 0.02)))
    else:
        loss = var_v * Decimal(str(random.uniform(0.1, 0.9)))
    actual_losses.append(Decimal(str(round(float(loss), 6))))

result = VaRBacktestSuite.run_all(var_forecasts, actual_losses)
check(f"Violations: {result.n_violations}/{result.n_observations}", result.n_observations == n_var)
check("Kupiec: p computed", len(str(result.kupiec_pvalue)) > 0)
check("DQ: stat computed", result.dq_statistic >= 0)
check("Berkowitz: LR computed", result.berkowitz_lr >= 0)

rolling = VaRBacktestSuite.rolling_traffic_light(var_forecasts, actual_losses, window_size=100, step=20)
check(f"Rolling: {len(rolling.windows)} windows", len(rolling.windows) > 0)
check(f"Stability: {round(float(rolling.stability_score), 3)} in [0,1]", 0 <= float(rolling.stability_score) <= 1)

edge_lr, edge_p, edge_pass = VaRBacktestSuite.kupiec_test(0, 100, Decimal("0.99"))
check("Edge: 0/100 violations -> not pass", edge_pass is False)

es_forecasts = [v * Decimal("1.5") for v in var_forecasts[:n_var]]
z_obs, p_val, z_pass = VaRBacktestSuite.acerbi_szekely_test(var_forecasts, actual_losses, es_forecasts)
check(f"Acerbi-Szekely: Z={round(z_obs, 3)}", isinstance(z_obs, (int, float)))

# -- Summary --
print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"{failed} CHECK(S) FAILED")
print("=" * 60)
