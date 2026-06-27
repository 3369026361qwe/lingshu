"""
因子多维验证脚本。

核心计算方法已模块化至 yinzi.factor_validator:
  - FactorValidator.compute_ic_decay() — 多周期 IC 衰减
  - FactorValidator.compute_factor_autocorr() — 因子自相关
  - FactorValidator.ic_stability() — 滚动 IC 稳定性

本脚本提供 CSV 数据加载 + 市场 regime 条件 IC + 综合评分报告。
"""
import csv, json, time, os, sys
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
from math import sqrt, isnan, isinf, log2
from statistics import mean, stdev

import numpy as np
from scipy.stats import spearmanr, pearsonr
from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
FORWARD_HORIZONS = [5, 10, 20, 40, 60]  # 多周期前瞻
MIN_STOCKS = 30
N_QUANTILES = 10
BASE = Path('E:/28721/lingshu/data')

# ═══════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════
print('=' * 80)
print('  灵枢量化系统 — 因子多维验证')
print('=' * 80)

print('\n[1/7] Loading data...')
t0 = time.time()

# Daily bars for close prices and market index
with open(BASE / 'hs800_daily_all.csv', 'r', encoding='utf-8-sig') as f:
    raw_rows = list(csv.DictReader(f))

all_dates = sorted(set(r['trade_date'] for r in raw_rows))

# Close price map
close_map = defaultdict(dict)
for r in raw_rows:
    close_map[r['ts_code']][r['trade_date']] = Decimal(r['close'])

# Market proxy: equally-weighted average return (or use HS300 index)
# Build daily market return series
market_rets = {}
for i in range(1, len(all_dates)):
    d_prev, d_curr = all_dates[i-1], all_dates[i]
    rets = []
    for code in close_map:
        p0 = close_map[code].get(d_prev)
        p1 = close_map[code].get(d_curr)
        if p0 and p1 and p0 > 0:
            rets.append(float((p1 - p0) / p0))
    if rets:
        market_rets[d_curr] = mean(rets)

# Load factor values from DB
from shujuku.session import SessionContext
from sqlalchemy import text

with SessionContext() as s:
    rows = s.execute(text(
        'SELECT code, trade_date, category, factor_name, raw_value FROM factor_value ORDER BY trade_date, factor_name, code'
    )).fetchall()

# Build: {trade_date: {factor_name: {code: value}}}
factor_data = defaultdict(lambda: defaultdict(dict))
for code, trade_date, category, factor_name, raw_value in rows:
    try:
        val = Decimal(str(raw_value))
        td = str(trade_date)
        factor_data[td][factor_name][code] = val
    except:
        pass

fv_dates = sorted(factor_data.keys())
factor_names = sorted(set(fn for d in factor_data for fn in factor_data[d]))

print(f'  Factor values: {len(rows):,} rows | {len(fv_dates)} dates | {len(factor_names)} factors')
print(f'  Load time: {time.time()-t0:.1f}s')

# ═══════════════════════════════════════════════════════════
# 2. Multi-horizon forward returns
# ═══════════════════════════════════════════════════════════
print('\n[2/7] Computing multi-horizon forward returns...')
t0 = time.time()

# {horizon: {date: {code: return}}}
forward_rets = {h: {} for h in FORWARD_HORIZONS}

for mdate in fv_dates:
    if mdate not in all_dates:
        continue
    mdi = all_dates.index(mdate)
    for h in FORWARD_HORIZONS:
        future_idx = min(mdi + h, len(all_dates) - 1)
        future_date = all_dates[future_idx]
        fr = {}
        for code in close_map:
            p0 = close_map[code].get(mdate)
            p1 = close_map[code].get(future_date)
            if p0 and p1 and p0 > 0:
                fr[code] = float((p1 - p0) / p0)
        forward_rets[h][mdate] = fr

print(f'  Horizons: {FORWARD_HORIZONS} | {len(fv_dates)} dates each ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 3. Multi-horizon IC Decay Analysis
# ═══════════════════════════════════════════════════════════
print('\n[3/7] Multi-horizon IC decay analysis...')
t0 = time.time()

ic_decay = {}  # {factor_name: {horizon: [ic_list]}}
for fname in factor_names:
    ic_decay[fname] = {h: [] for h in FORWARD_HORIZONS}

for mdate in fv_dates:
    for h in FORWARD_HORIZONS:
        if mdate not in forward_rets[h]:
            continue
        fr = forward_rets[h][mdate]
        for fname in factor_names:
            fv = factor_data[mdate].get(fname, {})
            common = set(fv.keys()) & set(fr.keys())
            pairs = [(float(fv[c]), fr[c]) for c in common
                     if not isnan(float(fv[c])) and not isinf(float(fv[c])) and abs(float(fv[c])) < 1e8]
            if len(pairs) < MIN_STOCKS:
                continue
            f_arr = np.array([p[0] for p in pairs])
            r_arr = np.array([p[1] for p in pairs])
            try:
                ric, _ = spearmanr(f_arr, r_arr)
                if not isnan(ric):
                    ic_decay[fname][h].append(ric)
            except:
                pass

# Compute mean IC per horizon per factor
ic_decay_summary = {}
for fname in factor_names:
    summary = {}
    for h in FORWARD_HORIZONS:
        ics = ic_decay[fname][h]
        if len(ics) >= 12:
            summary[h] = {
                'mean_ic': mean(ics),
                'std_ic': stdev(ics),
                'ir': mean(ics) / stdev(ics) if stdev(ics) > 0 else 0,
                't_stat': mean(ics) / (stdev(ics) / sqrt(len(ics))) if stdev(ics) > 0 else 0,
                'n': len(ics),
            }
    if summary:
        ic_decay_summary[fname] = summary

print(f'  Computed: {len(ic_decay_summary)} factors × {len(FORWARD_HORIZONS)} horizons ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 4. Market Regime Analysis
# ═══════════════════════════════════════════════════════════
print('\n[4/7] Market regime analysis...')
t0 = time.time()

# Classify each date as bull/bear/sideways based on 60-day rolling return
REGIME_LOOKBACK = 60
regime_labels = {}  # {date: 'bull' | 'bear' | 'sideways'}

for i, d in enumerate(all_dates):
    if i < REGIME_LOOKBACK:
        continue
    window_rets = [market_rets.get(all_dates[j], 0) for j in range(i - REGIME_LOOKBACK, i)]
    cum_ret = sum(window_rets)
    if cum_ret > 0.10:
        regime_labels[d] = 'bull'
    elif cum_ret < -0.10:
        regime_labels[d] = 'bear'
    else:
        regime_labels[d] = 'sideways'

regime_counts = {r: sum(1 for v in regime_labels.values() if v == r) for r in ['bull', 'bear', 'sideways']}
print(f'  Regime distribution: {regime_counts}')

# Compute IC by regime for each factor
H_MAIN = 20  # use 20d forward as main horizon
regime_ic = defaultdict(lambda: defaultdict(list))  # {factor: {regime: [ic]}}

for mdate in fv_dates:
    if mdate not in forward_rets[H_MAIN] or mdate not in regime_labels:
        continue
    regime = regime_labels[mdate]
    fr = forward_rets[H_MAIN][mdate]
    for fname in factor_names:
        fv = factor_data[mdate].get(fname, {})
        common = set(fv.keys()) & set(fr.keys())
        pairs = [(float(fv[c]), fr[c]) for c in common
                 if not isnan(float(fv[c])) and not isinf(float(fv[c])) and abs(float(fv[c])) < 1e8]
        if len(pairs) < MIN_STOCKS:
            continue
        f_arr = np.array([p[0] for p in pairs])
        r_arr = np.array([p[1] for p in pairs])
        try:
            ric, _ = spearmanr(f_arr, r_arr)
            if not isnan(ric):
                regime_ic[fname][regime].append(ric)
        except:
            pass

# Summarize
regime_summary = {}
for fname in factor_names:
    if fname not in regime_ic:
        continue
    summary = {}
    for regime in ['bull', 'bear', 'sideways']:
        ics = regime_ic[fname].get(regime, [])
        if len(ics) >= 10:
            summary[regime] = {
                'mean_ic': mean(ics),
                't_stat': mean(ics) / (stdev(ics) / sqrt(len(ics))) if len(ics) > 1 and stdev(ics) > 0 else 0,
                'n': len(ics),
            }
    if len(summary) >= 2:  # need at least 2 regimes
        regime_summary[fname] = summary

print(f'  Regime IC computed: {len(regime_summary)} factors ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 5. Factor Autocorrelation (Turnover Proxy)
# ═══════════════════════════════════════════════════════════
print('\n[5/7] Factor autocorrelation (turnover proxy)...')
t0 = time.time()

autocorr_summary = {}
for fname in factor_names:
    autocorrs = []
    for i in range(1, len(fv_dates)):
        d_prev, d_curr = fv_dates[i-1], fv_dates[i]
        fv_prev = factor_data[d_prev].get(fname, {})
        fv_curr = factor_data[d_curr].get(fname, {})
        common = set(fv_prev.keys()) & set(fv_curr.keys())
        if len(common) < MIN_STOCKS:
            continue
        fv1 = [float(fv_prev[c]) for c in common
               if not isnan(float(fv_prev[c])) and not isinf(float(fv_prev[c])) and abs(float(fv_prev[c])) < 1e8]
        fv2 = [float(fv_curr[c]) for c in common
               if not isnan(float(fv_curr[c])) and not isinf(float(fv_curr[c])) and abs(float(fv_curr[c])) < 1e8]
        # Match by index
        n = min(len(fv1), len(fv2))
        if n < MIN_STOCKS:
            continue
        try:
            rho, _ = spearmanr(fv1[:n], fv2[:n])
            if not isnan(rho):
                autocorrs.append(rho)
        except:
            pass
    if len(autocorrs) >= 12:
        mac = mean(autocorrs)
        autocorr_summary[fname] = {
            'mean_autocorr': mac,
            'implied_turnover': (1.0 - mac) / 2.0,  # approximate turnover
            'n': len(autocorrs),
        }

print(f'  Autocorrelation computed: {len(autocorr_summary)} factors ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 6. Monotonicity Test
# ═══════════════════════════════════════════════════════════
print('\n[6/7] Monotonicity test...')
t0 = time.time()

monotonicity_scores = {}
for fname in factor_names:
    mono_scores = []
    for mdate in fv_dates:
        if mdate not in forward_rets[H_MAIN]:
            continue
        fr = forward_rets[H_MAIN][mdate]
        fv = factor_data[mdate].get(fname, {})
        common = set(fv.keys()) & set(fr.keys())
        pairs = [(c, float(fv[c]), fr[c]) for c in common
                 if not isnan(float(fv[c])) and not isinf(float(fv[c])) and abs(float(fv[c])) < 1e8]
        if len(pairs) < N_QUANTILES * 3:
            continue
        pairs.sort(key=lambda x: x[1])
        n = len(pairs)
        gs = n // N_QUANTILES
        layer_rets = []
        for g in range(N_QUANTILES):
            start = g * gs
            end = start + gs if g < N_QUANTILES - 1 else n
            gp = pairs[start:end]
            if gp:
                layer_rets.append(mean([p[2] for p in gp]))
        if len(layer_rets) < N_QUANTILES:
            continue

        # Count monotonic increases (pairwise)
        monotonic_pairs = 0
        total_pairs = 0
        for i in range(len(layer_rets)):
            for j in range(i + 1, len(layer_rets)):
                total_pairs += 1
                if layer_rets[j] > layer_rets[i]:
                    monotonic_pairs += 1
        mono_scores.append(monotonic_pairs / total_pairs if total_pairs > 0 else 0.5)

    if len(mono_scores) >= 12:
        monotonicity_scores[fname] = {
            'mean_mono': mean(mono_scores),
            'std_mono': stdev(mono_scores) if len(mono_scores) > 1 else 0,
            'n': len(mono_scores),
        }

print(f'  Monotonicity computed: {len(monotonicity_scores)} factors ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 7. Comprehensive Validation Report
# ═══════════════════════════════════════════════════════════
print('\n[7/7] Generating validation report...\n')

# Scoring function
def validate_factor(fname):
    """Return dict with all validation metrics and pass/fail flags."""
    result = {'name': fname, 'checks': {}, 'warnings': [], 'score': 0, 'max_score': 0}

    # ---- Check 1: IC at 20d (weight: 30%) ----
    result['max_score'] += 30
    if fname in ic_decay_summary and 20 in ic_decay_summary[fname]:
        stats = ic_decay_summary[fname][20]
        ic_abs = abs(stats['mean_ic'])
        t_abs = abs(stats['t_stat'])
        result['checks']['ic_20d'] = f"|IC|={ic_abs:.4f}, |t|={t_abs:.1f}"

        if ic_abs > 0.05 and t_abs > 3.0:
            result['score'] += 30
            result['checks']['ic_20d'] += ' ✓ STRONG'
        elif ic_abs > 0.02 and t_abs > 2.0:
            result['score'] += 20
            result['checks']['ic_20d'] += ' ✓ PASS'
        elif ic_abs > 0.01:
            result['score'] += 10
            result['checks']['ic_20d'] += ' ⚠ WEAK'
            result['warnings'].append(f'IC_20d weak: |IC|={ic_abs:.4f}')
        else:
            result['checks']['ic_20d'] += ' ✗ FAIL'
            result['warnings'].append(f'IC_20d fail: |IC|={ic_abs:.4f}')
    else:
        result['checks']['ic_20d'] = 'NO DATA'

    # ---- Check 2: IC decay profile (weight: 20%) ----
    result['max_score'] += 20
    if fname in ic_decay_summary:
        # Good decay: IC peaks at medium horizon (20d or 40d) and decays at 60d
        ic_profile = {}
        for h in FORWARD_HORIZONS:
            if h in ic_decay_summary[fname]:
                ic_profile[h] = abs(ic_decay_summary[fname][h]['mean_ic'])

        if ic_profile:
            # Check peak horizon
            peak_h = max(ic_profile, key=ic_profile.get)
            ic_peak = ic_profile[peak_h]
            ic_5d = ic_profile.get(5, 0)
            ic_60d = ic_profile.get(60, ic_peak)

            result['checks']['ic_decay'] = f"peak@{peak_h}d={ic_peak:.4f}, 5d={ic_5d:.4f}, 60d={ic_60d:.4f}"

            # Score: factor should have reasonable persistence
            decay_ratio = ic_60d / ic_peak if ic_peak > 0 else 0
            if ic_peak > 0.03 and decay_ratio > 0.5:
                result['score'] += 20
                result['checks']['ic_decay'] += ' ✓ GOOD'
            elif ic_peak > 0.02 and decay_ratio > 0.3:
                result['score'] += 15
                result['checks']['ic_decay'] += ' ✓ OK'
            elif ic_peak > 0.01:
                result['score'] += 8
                result['checks']['ic_decay'] += ' ⚠ FAST_DECAY'
            else:
                result['score'] += 3
                result['checks']['ic_decay'] += ' ✗ NO_POWER'
        else:
            result['checks']['ic_decay'] = 'NO PROFILE'
    else:
        result['checks']['ic_decay'] = 'NO DATA'

    # ---- Check 3: Regime robustness (weight: 15%) ----
    result['max_score'] += 15
    if fname in regime_summary:
        regimes = regime_summary[fname]
        ic_by_regime = {r: regimes[r]['mean_ic'] for r in regimes}

        result['checks']['regime'] = ' | '.join(f"{r}: IC={ic_by_regime[r]:+.4f}" for r in ['bull', 'bear', 'sideways'] if r in ic_by_regime)

        # Score: consistency across regimes (low variance in IC sign)
        if len(ic_by_regime) >= 3:
            signs = [1 if v > 0 else -1 for v in ic_by_regime.values()]
            sign_consistent = len(set(signs)) == 1
            mag_variance = stdev(list(ic_by_regime.values())) if len(ic_by_regime) > 1 else 999

            if sign_consistent and mag_variance < 0.03:
                result['score'] += 15
                result['checks']['regime'] += ' ✓ ROBUST'
            elif sign_consistent:
                result['score'] += 10
                result['checks']['regime'] += ' ✓ STABLE_SIGN'
            else:
                result['score'] += 5
                result['checks']['regime'] += ' ⚠ UNSTABLE'
                result['warnings'].append('IC sign varies across regimes')
        elif len(ic_by_regime) >= 2:
            result['score'] += 8
            result['checks']['regime'] += ' ⚠ INCOMPLETE'
    else:
        result['checks']['regime'] = 'NO DATA'

    # ---- Check 4: Monotonicity (weight: 15%) ----
    result['max_score'] += 15
    if fname in monotonicity_scores:
        mono = monotonicity_scores[fname]['mean_mono']
        result['checks']['monotonicity'] = f"score={mono:.3f}"

        if mono > 0.75:
            result['score'] += 15
            result['checks']['monotonicity'] += ' ✓ STRONG'
        elif mono > 0.60:
            result['score'] += 12
            result['checks']['monotonicity'] += ' ✓ OK'
        elif mono > 0.50:
            result['score'] += 6
            result['checks']['monotonicity'] += ' ⚠ MARGINAL'
        else:
            result['checks']['monotonicity'] += ' ✗ NON-MONOTONIC'
            result['warnings'].append(f'Monotonicity low: {mono:.3f}')
    else:
        result['checks']['monotonicity'] = 'NO DATA'

    # ---- Check 5: Factor stability / turnover (weight: 20%) ----
    result['max_score'] += 20
    if fname in autocorr_summary:
        ac = autocorr_summary[fname]
        result['checks']['stability'] = f"autocorr={ac['mean_autocorr']:.3f}, turnover≈{ac['implied_turnover']:.2%}"

        if ac['mean_autocorr'] > 0.85:
            result['score'] += 20
            result['checks']['stability'] += ' ✓ STABLE'
        elif ac['mean_autocorr'] > 0.70:
            result['score'] += 15
            result['checks']['stability'] += ' ✓ OK'
        elif ac['mean_autocorr'] > 0.50:
            result['score'] += 8
            result['checks']['stability'] += ' ⚠ HIGH_TURNOVER'
            result['warnings'].append(f'High turnover: autocorr={ac["mean_autocorr"]:.3f}')
        else:
            result['score'] += 3
            result['checks']['stability'] += ' ✗ UNSTABLE'
            result['warnings'].append(f'Very unstable: autocorr={ac["mean_autocorr"]:.3f}')
    else:
        result['checks']['stability'] = 'NO DATA'

    # ---- Final grade ----
    score_pct = result['score'] / result['max_score'] * 100 if result['max_score'] > 0 else 0
    if score_pct >= 80:
        result['grade'] = 'PASS ✓'
        result['action'] = 'KEEP'
    elif score_pct >= 55:
        result['grade'] = 'CONDITIONAL △'
        result['action'] = 'REVIEW'
    else:
        result['grade'] = 'FAIL ✗'
        result['action'] = 'DROP'

    result['score_pct'] = score_pct
    return result

# Run validation for all factors
validation_results = {}
for fname in factor_names:
    validation_results[fname] = validate_factor(fname)

# ═══════════════════════════════════════════════════════════
# Print Report
# ═══════════════════════════════════════════════════════════

# Sort by score
sorted_results = sorted(validation_results.items(), key=lambda x: x[1]['score_pct'], reverse=True)

# Section A: Individual factor validation cards
print('─' * 80)
print('  因子验证卡片 (按综合得分排序)')
print('─' * 80)

for rank, (fname, vr) in enumerate(sorted_results, 1):
    sc = vr['score']
    ms = vr['max_score']
    pct = vr['score_pct']

    # Color indicator
    if vr['action'] == 'KEEP':
        icon = '🟢'
    elif vr['action'] == 'REVIEW':
        icon = '🟡'
    else:
        icon = '🔴'

    print(f'\n  {icon} [{rank:>2d}] {fname:25s}  Score: {sc}/{ms} ({pct:.0f}%)  → {vr["grade"]} ({vr["action"]})')

    for check_name, check_result in vr['checks'].items():
        print(f'      {check_name:15s}: {check_result}')

    if vr['warnings']:
        for w in vr['warnings']:
            print(f'      ⚠ {w}')

# Section B: Multi-horizon IC decay table
print(f'\n{"─"*80}')
print(f'  多周期 IC 衰减矩阵 (mean Rank IC)')
print(f'{"─"*80}')

header = f'  {"Factor":25s}'
for h in FORWARD_HORIZONS:
    header += f'  {h:>4d}d IC'
header += f'  {"Peak":>6s}  {"Decay":>6s}'
print(header)
print(f'  {"-"*72}')

for fname, _ in sorted_results:
    if fname not in ic_decay_summary:
        continue
    row = f'  {fname:25s}'
    ic_vals = {}
    for h in FORWARD_HORIZONS:
        if h in ic_decay_summary[fname]:
            ic_vals[h] = ic_decay_summary[fname][h]['mean_ic']
            row += f'  {ic_vals[h]:+7.4f}'
        else:
            row += f'  {"N/A":>7s}'

    if ic_vals:
        peak_h = max(ic_vals, key=lambda k: abs(ic_vals[k]))
        peak_ic = ic_vals[peak_h]
        decay_60 = ic_vals.get(60, peak_ic)
        decay_ratio = abs(decay_60 / peak_ic) if abs(peak_ic) > 0.001 else 0
        row += f'  {peak_h:>4d}d  {decay_ratio:>5.0%}'
    print(row)

# Section C: Regime IC comparison
print(f'\n{"─"*80}')
print(f'  市场 Regime 条件 IC (20d forward)')
print(f'{"─"*80}')
print(f'  {"Factor":25s}  {"Bull IC":>10s}  {"Bear IC":>10s}  {"Side IC":>10s}  {"Consistent":>10s}')
print(f'  {"-"*70}')

for fname, _ in sorted_results:
    if fname not in regime_summary:
        continue
    rs = regime_summary[fname]
    bull_ic = f'{rs["bull"]["mean_ic"]:+.4f}' if 'bull' in rs else 'N/A'
    bear_ic = f'{rs["bear"]["mean_ic"]:+.4f}' if 'bear' in rs else 'N/A'
    side_ic = f'{rs["sideways"]["mean_ic"]:+.4f}' if 'sideways' in rs else 'N/A'

    signs = []
    for r in ['bull', 'bear', 'sideways']:
        if r in rs:
            signs.append(1 if rs[r]['mean_ic'] > 0 else -1)
    consistent = 'YES' if len(set(signs)) <= 1 else 'NO'

    print(f'  {fname:25s}  {bull_ic:>10s}  {bear_ic:>10s}  {side_ic:>10s}  {consistent:>10s}')

# Section D: Factor stability ranking
print(f'\n{"─"*80}')
print(f'  因子稳定性 (自相关 → 低换手率)')
print(f'{"─"*80}')
stability_ranking = sorted(autocorr_summary.items(),
                           key=lambda x: x[1]['mean_autocorr'], reverse=True)
for rank, (fname, ac) in enumerate(stability_ranking[:15], 1):
    bar_len = int(ac['mean_autocorr'] * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    print(f'  {rank:>3d}. {fname:25s}  autcorr={ac["mean_autocorr"]:.3f}  '
          f'turnover≈{ac["implied_turnover"]:.1%}  [{bar}]')

# Section E: Monotonicity ranking
print(f'\n{"─"*80}')
print(f'  因子单调性 (10分位组收益单调递增率)')
print(f'{"─"*80}')
mono_ranking = sorted(monotonicity_scores.items(),
                      key=lambda x: x[1]['mean_mono'], reverse=True)
for rank, (fname, ms) in enumerate(mono_ranking, 1):
    bar_len = int(ms['mean_mono'] * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    print(f'  {rank:>3d}. {fname:25s}  mono={ms["mean_mono"]:.3f}±{ms["std_mono"]:.3f}  [{bar}]')

# Section F: Final summary
print(f'\n{"="*80}')
print(f'  因子验证总结')
print(f'{"="*80}')

keep_list = [fn for fn, vr in sorted_results if vr['action'] == 'KEEP']
review_list = [fn for fn, vr in sorted_results if vr['action'] == 'REVIEW']
drop_list = [fn for fn, vr in sorted_results if vr['action'] == 'DROP']

print(f'\n  🟢 通过验证 (KEEP):  {len(keep_list)} 个')
if keep_list:
    for fn in keep_list:
        vr = validation_results[fn]
        print(f'      {fn:25s}  {vr["score"]}/{vr["max_score"]} ({vr["score_pct"]:.0f}%)')

print(f'\n  🟡 条件通过 (REVIEW): {len(review_list)} 个')
if review_list:
    for fn in review_list:
        vr = validation_results[fn]
        print(f'      {fn:25s}  {vr["score"]}/{vr["max_score"]} ({vr["score_pct"]:.0f}%)')
        for w in vr['warnings']:
            print(f'        ⚠ {w}')

print(f'\n  🔴 未通过 (DROP):  {len(drop_list)} 个')
if drop_list:
    for fn in drop_list:
        vr = validation_results[fn]
        print(f'      {fn:25s}  {vr["score"]}/{vr["max_score"]} ({vr["score_pct"]:.0f}%)')
        for w in vr['warnings']:
            print(f'        ⚠ {w}')

print(f'\n  最终因子池: {len(keep_list)} 个核心因子')
print(f'  {", ".join(keep_list) if keep_list else "(none)"}')
print(f'{"="*80}')
