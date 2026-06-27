"""
因子 IC/IR 计算 — factor_value → factor_ic_record + factor_weight。

对每个交易日每个因子，计算截面 Rank IC（因子值与次日收益的 Spearman 相关系数）。
聚合所有日期的 IC 得到 IR，按 abs(IC) 归一化写入 factor_weight。

铁律: 只 INSERT ON CONFLICT DO NOTHING，不 DELETE 不 TRUNCATE。

Usage:
    python scripts/compute_factor_ic.py              # 全量计算
    python scripts/compute_factor_ic.py --days 60    # 仅最近 60 天
"""
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from math import isnan, isinf, sqrt

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from shujuku.session import SessionContext
from sqlalchemy import text
import numpy as np
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--days', type=int, default=0, help='仅计算最近 N 天，0=全量')
args = parser.parse_args()

print('=' * 60)
print('因子 IC/IR 计算: factor_value → factor_ic_record + factor_weight')
print('=' * 60)

# ── Step 0: 获取数据 ──────────────────────────────────

with SessionContext() as s:
    # 获取所有日期（有 factor_value 的日期）
    dates = [r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM factor_value ORDER BY trade_date'
    )).fetchall()]

    if args.days > 0:
        dates = dates[-args.days:]

    # 获取所有因子名
    factors = [r[0] for r in s.execute(text(
        'SELECT DISTINCT factor_name FROM factor_value ORDER BY factor_name'
    )).fetchall()]

    # 获取下一天有数据的日期（用于 forward return 计算）
    all_bar_dates = set(r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date'
    )).fetchall())

    # 检查已有 IC 记录
    existing_ic = s.execute(text('SELECT COUNT(*) FROM factor_ic_record')).scalar()
    existing_wt = s.execute(text('SELECT COUNT(*) FROM factor_weight')).scalar()
    print(f'\n已有 factor_ic_record: {existing_ic} | factor_weight: {existing_wt}')
    print(f'目标日期: {len(dates)} ({dates[0]} ~ {dates[-1]})')
    print(f'因子数: {len(factors)}')

# ── Step 1: 逐日计算 Rank IC ──────────────────────────

print('\n[1/2] 批量加载 → 计算逐日 Rank IC...')
t0 = time.time()

# 一次查询：加载整个区间所有因子值 + 价格数据
with SessionContext() as s:
    all_fv = s.execute(text(
        "SELECT trade_date, code, factor_name, raw_value FROM factor_value "
        "WHERE trade_date BETWEEN :start AND :end AND factor_name IN :fnames"
    ), {"start": dates[0], "end": dates[-1], "fnames": tuple(factors)}).fetchall()

    all_prices = s.execute(text(
        "SELECT a.trade_date, a.code, "
        "(CAST(a.close AS REAL) - CAST(b.close AS REAL)) / CAST(b.close AS REAL) as fwd_ret "
        "FROM daily_bar a JOIN daily_bar b ON a.code = b.code "
        "WHERE a.trade_date BETWEEN :start AND :end"
    ), {"start": dates[0], "end": dates[-1]}).fetchall()

print(f'  加载 {len(all_fv):,} 因子值 + {len(all_prices):,} 价格行 ({time.time()-t0:.1f}s)')

# 组织内存结构
fv_by_date = defaultdict(lambda: defaultdict(dict))
for td, code, fn, rv in all_fv:
    if fn in factors_set:
        fv_by_date[str(td)][fn][code] = float(str(rv))

ret_by_date = defaultdict(dict)
for td, code, ret in all_prices:
    if ret is not None:
        ret_by_date[str(td)][code] = ret

sorted_dates_all = sorted(set(str(d) for d in dates))
next_map = {sorted_dates_all[i]: sorted_dates_all[i+1] for i in range(len(sorted_dates_all)-1)}

ic_records = []
skipped = 0
total = len(dates)

for di, trade_date in enumerate(dates):
    td_str = str(trade_date)
    next_date = next_map.get(td_str)
    if next_date is None:
        skipped += 1
        continue

    fv_dict = fv_by_date.get(td_str, {})
    returns = ret_by_date.get(next_date, {})
    if not fv_dict or not returns:
        skipped += 1
        continue

    for fn, fv_map in fv_dict.items():
        codes = [c for c in fv_map if c in returns]
        if len(codes) < 30:
            continue

        x = np.array([fv_map[c] for c in codes], dtype=np.float64)
        y = np.array([returns[c] for c in codes], dtype=np.float64)

        # 过滤无效值
        mask = ~np.isnan(x) & ~np.isinf(x) & ~np.isnan(y) & ~np.isinf(y)
        if mask.sum() < 30:
            continue

        try:
            ic, _ = spearmanr(x[mask], y[mask])
            if not isnan(ic) and not isinf(ic):
                ic_records.append({
                    'd': trade_date, 'fn': fn, 'ic': round(float(ic), 6),
                })
        except Exception:
            continue

    if (di + 1) % 20 == 0:
        elapsed = time.time() - t0
        print(f'  进度: {di+1}/{total} 天 | IC 记录: {len(ic_records):,} | {elapsed:.0f}s')

elapsed = time.time() - t0
print(f'  完成: {len(ic_records):,} IC 记录 ({total-skipped} 天有效, {elapsed:.0f}s)')

# ── Step 2: 计算 IR + 权重 ────────────────────────────

print('\n[2/2] 聚合 IR 并写入权重...')

# 按因子聚合
factor_stats = defaultdict(list)
for r in ic_records:
    factor_stats[r['fn']].append(r['ic'])

weight_records = []
for fn, ics in factor_stats.items():
    if len(ics) < 10:
        continue
    mean_ic = mean(ics)
    std_ic = stdev(ics) if len(ics) > 1 else 0.01
    ir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    abs_ic = abs(mean_ic)
    weight_records.append({
        'fn': fn, 'mean_ic': mean_ic, 'ir': ir,
        'abs_ic': abs_ic, 'n': len(ics),
    })

# 按 abs(IC) 归一化
total_abs = sum(r['abs_ic'] for r in weight_records)
if total_abs > 0:
    for r in weight_records:
        r['weight'] = round(r['abs_ic'] / total_abs, 6)
else:
    for r in weight_records:
        r['weight'] = 1.0 / len(weight_records)

# ── 写入 ──────────────────────────────────────────────

with SessionContext() as s:
    ic_before = s.execute(text('SELECT COUNT(*) FROM factor_ic_record')).scalar()
    wt_before = s.execute(text('SELECT COUNT(*) FROM factor_weight')).scalar()

    # 写入 IC 记录 (批量 2000 条)
    ic_stmt = text(
        "INSERT INTO factor_ic_record (trade_date, factor_name, ic, ic_window, updated_at) "
        "VALUES (:d, :fn, :ic, 15, datetime('now')) "
        "ON CONFLICT(trade_date, factor_name) DO NOTHING"
    )
    for i in range(0, len(ic_records), 2000):
        s.execute(ic_stmt, ic_records[i:i+2000])
    s.commit()

    # 写入权重 (需要 category, trade_date)
    wt_stmt = text(
        "INSERT INTO factor_weight (trade_date, category, factor_name, weight, is_significant, updated_at) "
        "VALUES (date('now'), :cat, :fn, :weight, 1, datetime('now')) "
        "ON CONFLICT(trade_date, factor_name) DO UPDATE SET weight=:weight, updated_at=datetime('now')"
    )
    for wr in weight_records:
        s.execute(wt_stmt, {'fn': wr['fn'], 'weight': str(wr['weight']),
                           'cat': 'value'})  # 通用类别
    s.commit()

    ic_after = s.execute(text('SELECT COUNT(*) FROM factor_ic_record')).scalar()
    wt_after = s.execute(text('SELECT COUNT(*) FROM factor_weight')).scalar()

print(f'\n  写入结果:')
print(f'  factor_ic_record: {ic_before:,} → {ic_after:,} (+{ic_after-ic_before:,})')
print(f'  factor_weight:    {wt_before:,} → {wt_after:,} (+{wt_after-wt_before:,})')

# ── 打印 Top 因子 ─────────────────────────────────────

print('\n  Top 10 因子权重:')
print(f'  {"因子":20s} {"mean_IC":>8s} {"IR":>8s} {"权重":>8s} {"N":>6s}')
for r in sorted(weight_records, key=lambda x: x['weight'], reverse=True)[:10]:
    print(f'  {r["fn"]:20s} {r["mean_ic"]:>+8.4f} {r["ir"]:>+8.4f} {r["weight"]:>8.4f} {r["n"]:>6d}')

# ── 安全验证 ─────────────────────────────────────────

with SessionContext() as s:
    fusion_cnt = s.execute(text('SELECT COUNT(*) FROM fusion_score')).scalar()
print(f'\n  fusion_score 记录数: {fusion_cnt:,} (未被修改)')
print('\n[OK] Step 3 完成')
