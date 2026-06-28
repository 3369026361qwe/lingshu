"""全量数据入库 — 使用原始SQL批量插入，高性能。
安全说明: 本脚本仅管理 stock_info/daily_bar/financial_report 三张表，
使用定向 TRUNCATE 而非 drop_all，不会影响 factor_value 等计算密集表。
"""
import csv
import json
import os
import time
from pathlib import Path

import tushare as ts
from dotenv import load_dotenv

load_dotenv(Path('E:/28721/lingshu/.env'))
ts.set_token(os.environ['TUSHARE_TOKEN'])
pro = ts.pro_api()

from sqlalchemy import text

from shujuku.session import SessionContext, init_db

base = Path('E:/28721/lingshu/data')

# 安全初始化: create_all 但不 drop_all，保护 factor_value 等表
init_db(drop_all=False)
print('DB tables ensured')

# 定向清空本脚本管理的三张表（不影响 factor_value 等）
with SessionContext() as s:
    for table in ['daily_bar', 'financial_report', 'stock_info']:
        s.execute(text(f'DELETE FROM {table}'))
    s.commit()
print('Target tables cleaned (stock_info, daily_bar, financial_report)')

# Load HS800 codes
with open(base / 'hs800_daily_all.csv', encoding='utf-8-sig') as f:
    hs800_codes = set(r['ts_code'] for r in csv.DictReader(f))
print(f'HS800 stocks: {len(hs800_codes)}')

# ===== 1. Stock info =====
print('\n[1/4] Stock info...')
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
with SessionContext() as s:
    s.execute(text('BEGIN'))
    for _, row in df.iterrows():
        code = row['ts_code']
        if code not in hs800_codes:
            continue
        name = str(row.get('name','') or '')
        exchange = 'SH' if '.SH' in code else 'SZ'
        list_date = str(row.get('list_date','') or '')
        s.execute(text(
            "INSERT INTO stock_info (code, name, exchange, listing_date, is_active, updated_at) "
            "VALUES (:c, :n, :x, :d, 1, datetime('now')) "
            "ON CONFLICT(code) DO UPDATE SET name=:n, is_active=1, updated_at=datetime('now')"
        ), {'c': code, 'n': name, 'x': exchange, 'd': list_date})
    s.commit()
    result = s.execute(text('SELECT COUNT(*) FROM stock_info')).scalar()
print(f'  Stocks: {result}')

# ===== 2. Daily bars =====
print('\n[2/4] Daily bars (883K rows)...')
with open(base / 'hs800_daily_all.csv', encoding='utf-8-sig') as f:
    all_rows = list(csv.DictReader(f))

BATCH = 2000
bar_count = 0
t0 = time.time()

with SessionContext() as s:
    s.execute(text('BEGIN'))
    stmt = text(
        "INSERT INTO daily_bar (code, trade_date, open, high, low, close, volume, amount, turnover_rate, is_st, updated_at) "
        "VALUES (:c, :d, :o, :h, :l, :cl, :v, :a, :t, 0, datetime('now')) "
        "ON CONFLICT(code, trade_date) DO NOTHING"
    )
    batch = []
    for row in all_rows:
        batch.append({
            'c': row['ts_code'], 'd': row['trade_date'],
            'o': row['open'] or '0', 'h': row['high'] or '0',
            'l': row['low'] or '0', 'cl': row['close'] or '0',
            'v': row['vol'] or '0', 'a': row['amount'] or '0',
            't': row.get('turnover_rate', '0') or '0',
        })
        if len(batch) >= BATCH:
            s.execute(stmt, batch)
            bar_count += len(batch)
            batch = []
            if bar_count % 100000 == 0:
                print(f'  {bar_count:,}/{len(all_rows):,} ({time.time()-t0:.0f}s)')
    if batch:
        s.execute(stmt, batch)
        bar_count += len(batch)
    s.commit()
    result = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
print(f'  Daily bars: {result:,} ({time.time()-t0:.0f}s)')

# ===== 3. Financial reports =====
print('\n[3/4] Financial reports...')
with open(base / 'financial_data.json', encoding='utf-8') as f:
    fin_data = json.load(f)

fin_count = 0
with SessionContext() as s:
    s.execute(text('BEGIN'))
    stmt = text(
        "INSERT INTO financial_report (code, report_date, report_type, pe, pb, ps, roe, roa, gross_margin, net_margin, revenue, net_profit, operating_cashflow, updated_at) "
        "VALUES (:c, :d, :t, :pe, :pb, :ps, :roe, :roa, :gm, :nm, :rev, :np, :ocf, datetime('now')) "
        "ON CONFLICT(code, report_date) DO UPDATE SET pe=:pe, pb=:pb, ps=:ps, roe=:roe, roa=:roa, gross_margin=:gm, net_margin=:nm, revenue=:rev, net_profit=:np, operating_cashflow=:ocf, updated_at=datetime('now')"
    )
    batch = []
    for code, d in fin_data.items():
        if code not in hs800_codes:
            continue
        batch.append({
            'c': code, 'd': '20260606', 't': 'Q1',
            'pe': d.get('pe'), 'pb': d.get('pb'), 'ps': d.get('ps'),
            'roe': d.get('roe'), 'roa': d.get('roa'),
            'gm': d.get('gross_margin'), 'nm': d.get('net_margin'),
            'rev': d.get('revenue'), 'np': d.get('net_profit'),
            'ocf': d.get('operating_cashflow'),
        })
        if len(batch) >= 500:
            s.execute(stmt, batch)
            fin_count += len(batch)
            batch = []
    if batch:
        s.execute(stmt, batch)
        fin_count += len(batch)
    s.commit()
    result = s.execute(text('SELECT COUNT(*) FROM financial_report')).scalar()
print(f'  Financial reports: {result}')

# ===== 4. Verify =====
print('\n[4/4] Verification...')
with SessionContext() as s:
    si = s.execute(text('SELECT COUNT(*) FROM stock_info')).scalar()
    db = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
    fr = s.execute(text('SELECT COUNT(*) FROM financial_report')).scalar()
    # Show sample
    for table, count in [('stock_info', si), ('daily_bar', db), ('financial_report', fr)]:
        print(f'  {table}: {count:,} rows')
    # Sample query
    rows = s.execute(text("SELECT code, trade_date, close FROM daily_bar WHERE code='000001.SZ' ORDER BY trade_date DESC LIMIT 3")).fetchall()
    print('\n  Sample (000001.SZ):')
    for r in rows:
        print(f'    {r[0]} {r[1]} close={r[2]}')

db_path = Path('data/lingshu.db')
if db_path.exists():
    print(f'\nDB file: {db_path} ({db_path.stat().st_size/1024/1024:.0f} MB)')
print('=== Import complete ===')
