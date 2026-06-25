"""
全量因子计算脚本（优化版）。
覆盖全部 1312 个交易日，从 daily_bar 逐日计算因子值写入 factor_value 表。

性能优化:
  - CSV 一次性加载到内存，避免逐日 DB 查询
  - 每 10 天批量 commit，减少 I/O
  - 跳过已有数据的日期（支持增量）
  - 180 天回看窗口

目标: 35 因子 × 709 股票 × 1312 天 ≈ 3,260 万条记录
预计耗时: ~2 小时
"""
import csv, json, time, os, sys
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
from math import isnan, isinf

from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))

sys.stdout.reconfigure(encoding='utf-8')
base = Path('E:/28721/lingshu/data')

# ═══════════════════════════════════════════════════════════
# 1. Load all data into memory
# ═══════════════════════════════════════════════════════════
print('Loading daily data from CSV...')
t0 = time.time()

with open(base / 'hs800_daily_all.csv', 'r', encoding='utf-8-sig') as f:
    raw_rows = list(csv.DictReader(f))

# Build: {code: {trade_date: {bar_dict}}}
daily_map = defaultdict(dict)
for r in raw_rows:
    daily_map[r['ts_code']][r['trade_date']] = {
        'open': Decimal(r['open']), 'high': Decimal(r['high']),
        'low': Decimal(r['low']), 'close': Decimal(r['close']),
        'volume': Decimal(r['vol']), 'amount': Decimal(r['amount']),
        'turnover_rate': Decimal(r['turnover_rate'] or '0'),
    }

all_dates = sorted(set(r['trade_date'] for r in raw_rows))
codes_all = sorted(daily_map.keys())
print(f'  Loaded: {len(raw_rows):,} rows | {len(codes_all)} stocks | {len(all_dates)} dates ({all_dates[0]}~{all_dates[-1]})')
print(f'  Memory load: {time.time()-t0:.1f}s')

# ═══════════════════════════════════════════════════════════
# 2. Load financial data
# ═══════════════════════════════════════════════════════════
with open(base / 'financial_data.json', 'r', encoding='utf-8') as f:
    fin_raw = json.load(f)
fin_map = {}
for code, d in fin_raw.items():
    fin_map[code] = {k: Decimal(v) if v is not None else None for k, v in d.items()}

# ═══════════════════════════════════════════════════════════
# 3. Build factor engine
# ═══════════════════════════════════════════════════════════
from yinzi.engine import create_default_engine
engine = create_default_engine(max_workers=8)

# Register Alpha factors
from yinzi.alpha_factors import (ROCFactor, STDFactor, CORRFactor, MAXFactor, MINFactor,
    VMAFactor, VSTDFactor, CNTPFactor, BETAFactor, RSQRFactor, RANKFactor,
    SKEWFactor, KURTFactor, TURNFactor, AMPFactor, VWAPFactor, HLSpreadFactor, OCFactor)
alpha_cls = [ROCFactor, STDFactor, CORRFactor, MAXFactor, MINFactor, VMAFactor, VSTDFactor,
             CNTPFactor, BETAFactor, RSQRFactor, RANKFactor, SKEWFactor, KURTFactor,
             TURNFactor, AMPFactor, VWAPFactor, HLSpreadFactor, OCFactor]
for cls in alpha_cls:
    if hasattr(cls, 'WINDOWS'):
        for w in cls.WINDOWS:
            try: engine.register(cls(window=w))
            except: pass
    else:
        try: engine.register(cls())
        except: pass
print(f'Factors registered: {engine.factor_count}')

# ═══════════════════════════════════════════════════════════
# 4. Identify target dates (skip already-computed)
# ═══════════════════════════════════════════════════════════
from shujuku.session import SessionContext
from sqlalchemy import text

LOOKBACK = 180  # 回看窗口天数

with SessionContext() as s:
    existing_dates = set()
    rows = s.execute(text('SELECT DISTINCT trade_date FROM factor_value')).fetchall()
    for r in rows:
        existing_dates.add(r[0])
    print(f'Already computed: {len(existing_dates)} dates')

# Dates we can compute (need LOOKBACK days of history)
target_dates = [d for d in all_dates if all_dates.index(d) >= LOOKBACK]
print(f'Target dates: {len(target_dates)} (from {target_dates[0]} to {target_dates[-1]})')

# Filter out already-computed
target_dates = [d for d in target_dates if d not in existing_dates]
print(f'Remaining to compute: {len(target_dates)} dates')

if not target_dates:
    print('All dates already computed!')
    sys.exit(0)

# ═══════════════════════════════════════════════════════════
# 5. Main computation loop
# ═══════════════════════════════════════════════════════════
COMMIT_EVERY = 10  # commit every N days
total_inserted = 0
total_dates_done = 0
t_start = time.time()

print(f'\n{"="*60}')
print(f'  Full Factor Computation')
print(f'  {len(target_dates)} dates × ~{len(codes_all)} stocks × {engine.factor_count} factors')
print(f'  Estimated: ~{len(target_dates) * engine.factor_count * len(codes_all) / 1e6:.0f}M rows')
print(f'{"="*60}\n')

with SessionContext() as s:
    stmt = text(
        "INSERT INTO factor_value (code, trade_date, category, factor_name, raw_value, updated_at) "
        "VALUES (:c, :d, :cat, :n, :v, datetime('now')) "
        "ON CONFLICT(code, trade_date, factor_name) DO UPDATE SET raw_value=:v, updated_at=datetime('now')"
    )
    batch = []
    date_times = []

    for di, target_date in enumerate(target_dates):
        dt0 = time.time()

        # Get index of target_date
        tdi = all_dates.index(target_date)

        # Build lookback window
        window_dates = set(all_dates[max(0, tdi - LOOKBACK):tdi + 1])

        # Build per-stock daily data for this window
        dm = {}
        codes_active = []
        for code in codes_all:
            code_data = daily_map.get(code, {})
            code_window = {}
            for d in code_data:
                if d in window_dates:
                    code_window[d] = code_data[d]
            if len(code_window) >= 20:  # need min 20 days
                dm[code] = code_window
                codes_active.append(code)

        if len(codes_active) < 50:
            continue  # not enough stocks

        # Compute factors
        try:
            results = engine.compute_all(codes_active, dm, fin_map, parallel=True)
        except Exception as e:
            print(f'  ERROR {target_date}: {e}')
            continue

        # Build batch insert rows
        day_count = 0
        for r in results:
            if r.raw_value is not None:
                try:
                    fv = float(r.raw_value)
                    if isnan(fv) or isinf(fv) or abs(fv) >= 1e8:
                        continue
                    batch.append({
                        'c': r.code, 'd': target_date,
                        'cat': r.category.value, 'n': r.factor_name,
                        'v': str(r.raw_value),
                    })
                    day_count += 1
                except Exception:
                    pass

        total_inserted += day_count
        total_dates_done += 1
        date_times.append(time.time() - dt0)

        # Commit every N days
        if (di + 1) % COMMIT_EVERY == 0:
            if batch:
                s.execute(stmt, batch)
                s.commit()
                batch = []

            elapsed = time.time() - t_start
            avg_time = sum(date_times[-COMMIT_EVERY:]) / len(date_times[-COMMIT_EVERY:])
            remaining = (len(target_dates) - total_dates_done) * avg_time / 60
            speed = total_inserted / elapsed if elapsed > 0 else 0
            print(f'  [{di+1}/{len(target_dates)}] {target_date} '
                  f'| {total_inserted:,} rows ({total_inserted/1e6:.1f}M) '
                  f'| {elapsed/60:.0f}m elapsed | ~{remaining:.0f}m remaining '
                  f'| {speed:,.0f} rows/s')

    # Final commit
    if batch:
        s.execute(stmt, batch)
        s.commit()

    # Verify
    final_count = s.execute(text('SELECT COUNT(*) FROM factor_value')).scalar()
    final_dates = s.execute(text('SELECT COUNT(DISTINCT trade_date) FROM factor_value')).scalar()
    min_d = s.execute(text('SELECT MIN(trade_date) FROM factor_value')).scalar()
    max_d = s.execute(text('SELECT MAX(trade_date) FROM factor_value')).scalar()

total_elapsed = time.time() - t_start
print(f'\n{"="*60}')
print(f'  Factor Computation Complete')
print(f'  {"="*60}')
print(f'  Dates processed:   {total_dates_done}')
print(f'  Total rows:        {final_count:,} ({final_count/1e6:.1f}M)')
print(f'  Date range:        {min_d} ~ {max_d}')
print(f'  Unique dates:      {final_dates}')
print(f'  Total time:        {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)')
print(f'  Avg per day:       {total_elapsed/total_dates_done:.2f}s' if total_dates_done > 0 else '')
print(f'{"="*60}')
