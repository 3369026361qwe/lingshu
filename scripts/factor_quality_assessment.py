"""
因子质量全面评估脚本。
从 factor_value 表读取全量因子数据，进行多维度质量评估。

评估维度:
  1. Rank IC — Spearman秩相关系数（因子值 vs 未来收益）
  2. Pearson IC — Pearson线性相关系数
  3. IR (Information Ratio) — IC均值/IC标准差
  4. 分层回测 — 10分组Top-Bottom Spread
  5. NDCG@30 — 排序质量指标
  6. 因子相关性矩阵 — 识别冗余因子
  7. 滚动IC稳定性 — 60期滚动IC均值/标准差
  8. 因子分级 — A/B/C/D 综合评分

输出: IC/IR排名表 + 相关性矩阵 + 分层回测结果 + 综合报告
"""
import csv, json, time, os, sys
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
from math import sqrt, isnan, isinf, log2
from statistics import mean, stdev

import numpy as np
from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
FORWARD_DAYS = 20     # 前瞻收益率窗口
MIN_STOCKS = 30        # 最少股票数
MIN_IC_PERIODS = 12    # 最少IC期数（用于IR计算）
ROLLING_WINDOW = 60    # 滚动IC窗口

BASE = Path('E:/28721/lingshu/data')

# ═══════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════
print('Loading data...')
t0 = time.time()

# Load daily close prices for forward return computation
with open(BASE / 'hs800_daily_all.csv', 'r', encoding='utf-8-sig') as f:
    raw_rows = list(csv.DictReader(f))

all_dates = sorted(set(r['trade_date'] for r in raw_rows))

# Build close price map: {code: {date: close}}
close_map = defaultdict(dict)
for r in raw_rows:
    close_map[r['ts_code']][r['trade_date']] = Decimal(r['close'])

# Load factor values from DB
from shujuku.session import SessionContext
from sqlalchemy import text

with SessionContext() as s:
    rows = s.execute(text(
        'SELECT code, trade_date, category, factor_name, raw_value FROM factor_value ORDER BY trade_date, factor_name, code'
    )).fetchall()

print(f'  Factor values: {len(rows):,} rows')
print(f'  Load time: {time.time()-t0:.1f}s')

# Build: {trade_date: {factor_name: {code: value}}}
t0 = time.time()
factor_data = defaultdict(lambda: defaultdict(dict))
for code, trade_date, category, factor_name, raw_value in rows:
    try:
        val = Decimal(str(raw_value))
        if val is not None:
            td = str(trade_date)  # SQLite returns int for DATE, convert to string
            factor_data[td][factor_name][code] = val
    except:
        pass

# Sort dates
fv_dates = sorted(factor_data.keys())
factor_names = sorted(set(fn for d in factor_data for fn in factor_data[d]))
print(f'  Organized: {len(fv_dates)} dates × {len(factor_names)} factors')
print(f'  Build time: {time.time()-t0:.1f}s')

# ═══════════════════════════════════════════════════════════
# 2. Compute forward returns
# ═══════════════════════════════════════════════════════════
print('\nComputing forward returns...')
t0 = time.time()

# For each measurement date, compute FORWARD_DAYS forward return
forward_returns = {}  # {date: {code: return}}
for di, mdate in enumerate(fv_dates):
    mdi = all_dates.index(mdate) if mdate in all_dates else -1
    if mdi < 0:
        continue
    future_idx = min(mdi + FORWARD_DAYS, len(all_dates) - 1)
    future_date = all_dates[future_idx]

    fr = {}
    for code in close_map:
        p0 = close_map[code].get(mdate)
        p1 = close_map[code].get(future_date)
        if p0 and p1 and p0 > 0:
            fr[code] = (p1 - p0) / p0
    forward_returns[mdate] = fr

print(f'  Forward returns: {len(forward_returns)} dates ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 3. IC computation (Rank + Pearson)
# ═══════════════════════════════════════════════════════════
print('\nComputing IC series...')
t0 = time.time()

from scipy.stats import spearmanr, pearsonr

ic_data = defaultdict(lambda: {'rank_ic': [], 'pearson_ic': [], 'dates': [], 'n_stocks': []})

for mdate in fv_dates:
    if mdate not in forward_returns:
        continue
    fr = forward_returns[mdate]

    for fname in factor_names:
        fv = factor_data[mdate].get(fname, {})
        common = set(fv.keys()) & set(fr.keys())
        if len(common) < MIN_STOCKS:
            continue

        f_vals = [float(fv[c]) for c in common]
        r_vals = [float(fr[c]) for c in common]

        # Filter inf/nan
        valid = [(f, r) for f, r in zip(f_vals, r_vals)
                 if not isnan(f) and not isinf(f) and abs(f) < 1e8
                 and not isnan(r) and not isinf(r)]
        if len(valid) < MIN_STOCKS:
            continue

        f_arr = np.array([v[0] for v in valid])
        r_arr = np.array([v[1] for v in valid])

        try:
            ric, _ = spearmanr(f_arr, r_arr)
            if not isnan(ric):
                ic_data[fname]['rank_ic'].append(Decimal(str(round(ric, 6))))
        except:
            pass

        try:
            pic, _ = pearsonr(f_arr, r_arr)
            if not isnan(pic):
                ic_data[fname]['pearson_ic'].append(Decimal(str(round(pic, 6))))
        except:
            pass

        ic_data[fname]['dates'].append(mdate)
        ic_data[fname]['n_stocks'].append(len(valid))

f_count = sum(1 for f in ic_data if ic_data[f]['rank_ic'])
print(f'  IC computed: {f_count:,} factors ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 4. IR and Ranking
# ═══════════════════════════════════════════════════════════
print('\nComputing IR and rankings...')

factor_stats = []
for fname, data in ic_data.items():
    ric_list = data['rank_ic']
    if len(ric_list) < MIN_IC_PERIODS:
        continue

    ric_vals = [float(v) for v in ric_list]
    mic = mean(ric_vals)
    sic = stdev(ric_vals) if len(ric_vals) > 1 else 0.01
    ir = mic / sic if sic > 0 else 0

    pic_list = data['pearson_ic']
    mpic = mean([float(v) for v in pic_list]) if pic_list else 0

    # t-statistic
    t_stat = mic / (sic / sqrt(len(ric_vals))) if sic > 0 else 0

    # Win rate (% of periods with positive IC)
    pos_rate = sum(1 for v in ric_vals if v > 0) / len(ric_vals) * 100

    # IC stability (rolling 60-period)
    rolling_ics = []
    rolling_stds = []
    for i in range(ROLLING_WINDOW, len(ric_vals) + 1):
        window = ric_vals[i - ROLLING_WINDOW:i]
        rolling_ics.append(mean(window))
        rolling_stds.append(stdev(window) if len(window) > 1 else 0)
    ic_stability = stdev(rolling_ics) if len(rolling_ics) > 1 else 0  # lower = more stable

    factor_stats.append({
        'factor_name': fname,
        'n_periods': len(ric_vals),
        'mean_rank_ic': mic,
        'std_rank_ic': sic,
        'ir': ir,
        't_stat': t_stat,
        'mean_pearson_ic': mpic,
        'positive_rate': pos_rate,
        'ic_stability': ic_stability,
        'avg_n_stocks': mean(data['n_stocks']) if data['n_stocks'] else 0,
    })

# Sort by |IR|
factor_stats.sort(key=lambda x: abs(x['ir']), reverse=True)

# ═══════════════════════════════════════════════════════════
# 5. Factor correlation matrix
# ═══════════════════════════════════════════════════════════
print('Computing factor correlation matrix...')
t0 = time.time()

# Build factor return series: for each factor, compute cross-sectional mean per date
top_factors = [f['factor_name'] for f in factor_stats[:20]]  # top 20
factor_series = {fname: [] for fname in top_factors}
common_dates = []

for mdate in fv_dates:
    has_all = True
    row = {}
    for fname in top_factors:
        fv = factor_data[mdate].get(fname, {})
        vals = [float(v) for v in fv.values()
                if not isnan(float(v)) and not isinf(float(v)) and abs(float(v)) < 1e8]
        if len(vals) < MIN_STOCKS:
            has_all = False
            break
        row[fname] = mean(vals)
    if has_all:
        common_dates.append(mdate)
        for fname in top_factors:
            factor_series[fname].append(row[fname])

# Compute correlation
n_top = len(top_factors)
corr_matrix = np.zeros((n_top, n_top))
for i in range(n_top):
    for j in range(n_top):
        if i == j:
            corr_matrix[i, j] = 1.0
        else:
            si = np.array(factor_series[top_factors[i]])
            sj = np.array(factor_series[top_factors[j]])
            n = min(len(si), len(sj))
            if n > 10:
                c = np.corrcoef(si[:n], sj[:n])[0, 1]
                corr_matrix[i, j] = c if not isnan(c) else 0
            else:
                corr_matrix[i, j] = 0

print(f'  Correlation matrix: {n_top}×{n_top} over {len(common_dates)} dates ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 6. Layered backtest (quantile analysis)
# ═══════════════════════════════════════════════════════════
print('Computing layered backtest (quantile analysis)...')
t0 = time.time()

N_QUANTILES = 10
layered_results = {}

for fname in top_factors:
    all_spreads = []
    all_layer_rets = defaultdict(list)

    for mdate in fv_dates:
        if mdate not in forward_returns:
            continue
        fr = forward_returns[mdate]
        fv = factor_data[mdate].get(fname, {})

        common = set(fv.keys()) & set(fr.keys())
        pairs = [(c, float(fv[c]), float(fr[c])) for c in common
                 if not isnan(float(fv[c])) and not isinf(float(fv[c])) and abs(float(fv[c])) < 1e8]
        if len(pairs) < N_QUANTILES * 3:
            continue

        pairs.sort(key=lambda x: x[1])  # sort by factor value
        n = len(pairs)
        group_size = n // N_QUANTILES

        for g in range(N_QUANTILES):
            start = g * group_size
            end = start + group_size if g < N_QUANTILES - 1 else n
            gp = pairs[start:end]
            if gp:
                avg_ret = mean([p[2] for p in gp])
                all_layer_rets[g].append(avg_ret)

        # Top-bottom spread
        if all_layer_rets[0] and all_layer_rets[N_QUANTILES - 1]:
            spread = mean(all_layer_rets[N_QUANTILES - 1]) - mean(all_layer_rets[0])
            all_spreads.append(spread)

    if all_spreads:
        layered_results[fname] = {
            'mean_spread': mean(all_spreads),
            'spread_t': mean(all_spreads) / (stdev(all_spreads) / sqrt(len(all_spreads))) if len(all_spreads) > 1 and stdev(all_spreads) > 0 else 0,
            'n_periods': len(all_spreads),
            'layer_returns': {g: mean(rets) for g, rets in all_layer_rets.items()},
            'positive_spread_pct': sum(1 for s in all_spreads if s > 0) / len(all_spreads) * 100,
            'top_return': mean(all_layer_rets.get(N_QUANTILES - 1, [0])),
            'bottom_return': mean(all_layer_rets.get(0, [0])),
        }

print(f'  Layered backtest: {len(layered_results)} factors ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════════════
# 7. Grade assignment
# ═══════════════════════════════════════════════════════════
def assign_grade(ir, t_stat, pos_rate, spread_t):
    """综合评分: A (优秀) / B (良好) / C (一般) / D (差)"""
    score = 0
    if abs(ir) > 0.5: score += 3
    elif abs(ir) > 0.3: score += 2
    elif abs(ir) > 0.15: score += 1

    if abs(t_stat) > 2.0: score += 2
    elif abs(t_stat) > 1.0: score += 1

    if pos_rate > 60: score += 1
    elif pos_rate > 50: score += 0.5

    if abs(spread_t) > 2.0: score += 2
    elif abs(spread_t) > 1.0: score += 1

    if score >= 6: return 'A'
    elif score >= 4: return 'B'
    elif score >= 2: return 'C'
    return 'D'

# ═══════════════════════════════════════════════════════════
# 8. Print Report
# ═══════════════════════════════════════════════════════════
print(f'\n{"="*90}')
print(f'  灵枢量化系统 — 因子质量评估报告')
print(f'  评估区间: {fv_dates[0]} ~ {fv_dates[-1]} ({len(fv_dates)} 期)')
print(f'  前瞻窗口: {FORWARD_DAYS} 交易日')
print(f'  评估因子: {len(factor_stats)} 个')
print(f'{"="*90}')

# IC/IR Table
print(f'\n{"─"*90}')
print(f'  IC/IR 排名')
print(f'{"─"*90}')
header = f'  {"Rank":>4s}  {"Factor":25s}  {"N":>5s}  {"RankIC":>8s}  {"StdIC":>8s}  {"IR":>8s}  {"t-stat":>7s}  {"Win%":>6s}  {"Pearson":>8s}  {"Grade":>5s}'
print(header)
print(f'  {"-"*86}')

grades_dist = defaultdict(list)

for rank, fs in enumerate(factor_stats, 1):
    fname = fs['factor_name']
    spread_t = layered_results.get(fname, {}).get('spread_t', 0) if fname in layered_results else 0
    grade = assign_grade(fs['ir'], fs['t_stat'], fs['positive_rate'], spread_t)
    grades_dist[grade].append(fname)

    print(f'  {rank:>4d}  {fname:25s}  {fs["n_periods"]:>5d}  '
          f'{fs["mean_rank_ic"]:>+8.4f}  {fs["std_rank_ic"]:>8.4f}  '
          f'{fs["ir"]:>+8.4f}  {fs["t_stat"]:>+7.2f}  '
          f'{fs["positive_rate"]:>5.1f}%  {fs["mean_pearson_ic"]:>+8.4f}  {grade:>5s}')

# Grade distribution
print(f'\n{"─"*90}')
print(f'  因子分级分布')
print(f'{"─"*90}')
for g in ['A', 'B', 'C', 'D']:
    if g in grades_dist:
        factors = grades_dist[g]
        print(f'  {g} 级 ({len(factors)} 个): {", ".join(factors)}')

# Factor Correlation Summary
print(f'\n{"─"*90}')
print(f'  因子相关性矩阵 (Top 20, |corr| > 0.7 高亮)')
print(f'{"─"*90}')

# Print correlation matrix header
print(f'  {"":25s}', end='')
for i, fn in enumerate(top_factors):
    short = fn[:6]
    print(f'{short:>7s}', end='')
print()

for i, fni in enumerate(top_factors):
    print(f'  {fni:25s}', end='')
    for j, fnj in enumerate(top_factors):
        corr = corr_matrix[i, j]
        if i == j:
            print(f'  {"·":>5s}', end='')
        elif abs(corr) > 0.7:
            print(f'  \033[91m{corr:+.2f}\033[0m', end='')  # red for high corr
        else:
            print(f'  {corr:+.2f}', end='')
    print()

# Identify redundant factor pairs
print(f'\n  高相关因子对 (|corr| > 0.7):')
redundant_pairs = []
for i in range(n_top):
    for j in range(i + 1, n_top):
        if abs(corr_matrix[i, j]) > 0.7:
            redundant_pairs.append((top_factors[i], top_factors[j], corr_matrix[i, j]))
if redundant_pairs:
    for fi, fj, c in sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f'    {fi:25s} <-> {fj:25s}  corr={c:+.3f}')
else:
    print(f'    (none)')

# Layered Backtest Summary
print(f'\n{"─"*90}')
print(f'  分层回测 Top-Bottom Spread (10 分位组)')
print(f'{"─"*90}')
print(f'  {"Factor":25s}  {"Spread":>10s}  {"t-stat":>8s}  {"+Spread%":>8s}  {"Top Ret":>10s}  {"Bot Ret":>10s}')
print(f'  {"-"*78}')

sorted_layers = sorted(layered_results.items(),
                       key=lambda x: abs(x[1]['mean_spread']), reverse=True)
for fname, lr in sorted_layers:
    print(f'  {fname:25s}  {lr["mean_spread"]:>+10.4f}%  {lr["spread_t"]:>+8.2f}  '
          f'{lr["positive_spread_pct"]:>7.1f}%  {lr["top_return"]:>+10.4f}%  {lr["bottom_return"]:>+10.4f}%')

# NDCG and Ranking Metrics
print(f'\n{"─"*90}')
print(f'  排序质量指标 (NDCG@30)')
print(f'{"─"*90}')
from yinzi.factor_validator import FactorValidator
validator = FactorValidator()

ndcg_scores = {}
for fname in top_factors[:15]:
    ndcg_list = []
    for mdate in fv_dates:
        if mdate not in forward_returns:
            continue
        fr = forward_returns[mdate]
        fv = factor_data[mdate].get(fname, {})
        # Convert to Decimal
        fv_dec = {c: Decimal(str(v)) if isinstance(v, (int, float)) else v for c, v in fv.items()}
        fr_dec = {c: v for c, v in fr.items()}  # already Decimal
        ndcg = validator.compute_ndcg(fv_dec, fr_dec, k=30)
        if ndcg is not None:
            ndcg_list.append(float(ndcg))
    if ndcg_list:
        ndcg_scores[fname] = mean(ndcg_list)

for rank, (fname, ndcg) in enumerate(sorted(ndcg_scores.items(), key=lambda x: x[1], reverse=True), 1):
    print(f'  {rank:>3d}. {fname:25s}  NDCG@30 = {ndcg:.4f}')

# Rolling IC Stability
print(f'\n{"─"*90}')
print(f'  IC 稳定性 (60 期滚动标准差，越小越稳定)')
print(f'{"─"*90}')
stability_ranking = sorted(factor_stats, key=lambda x: x['ic_stability'])[:10]
for rank, fs in enumerate(stability_ranking, 1):
    print(f'  {rank:>3d}. {fs["factor_name"]:25s}  IC_stability={fs["ic_stability"]:.4f}  '
          f'IR={fs["ir"]:+.4f}  t={fs["t_stat"]:+.2f}')

# Final summary
print(f'\n{"="*90}')
print(f'  综合评估结论')
print(f'{"="*90}')
print(f'  有效因子 (|IR| > 0.3): {len(grades_dist.get("A", [])) + len(grades_dist.get("B", []))} 个')
print(f'  需优化因子 (0.15 < |IR| < 0.3): {len(grades_dist.get("C", []))} 个')
print(f'  无效因子 (|IR| < 0.15): {len(grades_dist.get("D", []))} 个')
print(f'  高相关因子对 (|corr| > 0.7): {len(redundant_pairs)} 对')
print(f'')
print(f'  建议:')
if len(grades_dist.get('D', [])) > 0:
    print(f'    1. 淘汰 D 级因子: {", ".join(grades_dist["D"])}')
if redundant_pairs:
    print(f'    2. 去冗余: 对高相关因子对，保留 |IR| 更高的那个')
a_plus_b = grades_dist.get('A', []) + grades_dist.get('B', [])
if a_plus_b:
    print(f'    3. 核心因子池: {", ".join(a_plus_b)} (A+B 级)')
print(f'{"="*90}')
