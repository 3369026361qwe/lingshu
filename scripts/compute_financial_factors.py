"""
从 financial_report 提取质量/估值因子 → factor_value。
铁律: INSERT ON CONFLICT DO NOTHING，绝不删除已有数据。

Usage:
    python scripts/compute_financial_factors.py
"""
import sys, os, time
from pathlib import Path

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from shujuku.session import SessionContext
from sqlalchemy import text

print('=' * 60)
print('财务因子提取: financial_report → factor_value')
print('=' * 60)

# 因子: (column_name, factor_name, category, direction)
# direction: +1 = 值越大越好, -1 = 值越小越好
FINANCIAL_FACTORS = [
    ('pe', 'pe', 'value', -1),
    ('pb', 'pb', 'value', -1),
    ('ps', 'ps', 'value', -1),
    ('roe', 'roe', 'quality', +1),
    ('roa', 'roa', 'quality', +1),
    ('gross_margin', 'gross_margin', 'quality', +1),
    ('net_margin', 'net_margin', 'quality', +1),
    ('free_cashflow_yield', 'free_cashflow_yield', 'quality', +1),
]

with SessionContext() as s:
    # 获取所有 trade_date
    dates = [r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM factor_value ORDER BY trade_date'
    )).fetchall()]
    print(f'目标日期: {len(dates)} ({dates[0]} ~ {dates[-1]})')

    # 获取每只股票的最新财务数据
    fin_rows = s.execute(text("""
        SELECT code, pe, pb, ps, roe, roa, gross_margin, net_margin, free_cashflow_yield
        FROM financial_report
        ORDER BY code, report_date DESC
    """)).fetchall()

    # 每只股票只取最新一期
    fin_map = {}
    for row in fin_rows:
        code = row[0]
        if code not in fin_map:
            vals = {}
            for i, (col, fn, cat, direction) in enumerate(FINANCIAL_FACTORS):
                raw = row[i + 1]
                if raw is not None:
                    try:
                        v = float(str(raw))
                        if v != v or abs(v) > 1e8:  # NaN or too large
                            continue
                        vals[fn] = (v, cat)
                    except (ValueError, TypeError):
                        continue
            if vals:
                fin_map[code] = vals

    print(f'有效财务数据: {len(fin_map)} 只股票')
    print(f'因子: {[f[1] for f in FINANCIAL_FACTORS]}')

    # 对于每个 trade_date，为有财务数据的股票创建因子
    before = s.execute(text('SELECT COUNT(*) FROM factor_value')).scalar()
    print(f'factor_value 操作前: {before:,}')

    batch_size = 0
    t0 = time.time()

    for di, trade_date in enumerate(dates):
        batch = []
        for code, facts in fin_map.items():
            for fn, (val, cat) in facts.items():
                batch.append({
                    'c': code, 'd': trade_date, 'cat': cat, 'fn': fn,
                    'rv': str(val), 'zs': str(val),
                })

        if batch:
            stmt = text(
                "INSERT INTO factor_value (code, trade_date, category, factor_name, raw_value, z_score, updated_at) "
                "VALUES (:c, :d, :cat, :fn, :rv, :zs, datetime('now')) "
                "ON CONFLICT(code, trade_date, factor_name) DO NOTHING"
            )
            for i in range(0, len(batch), 2000):
                s.execute(stmt, batch[i:i + 2000])
            s.commit()
            batch_size += len(batch)

        if (di + 1) % 30 == 0:
            print(f'  进度: {di+1}/{len(dates)} ({time.time()-t0:.0f}s)')

    after = s.execute(text('SELECT COUNT(*) FROM factor_value')).scalar()
    print(f'factor_value: {before:,} → {after:,} (+{after-before:,})')
    print(f'耗时: {time.time()-t0:.0f}s')

    # 验证: fusion_score 未变
    fs_cnt = s.execute(text('SELECT COUNT(*) FROM fusion_score')).scalar()
    print(f'fusion_score: {fs_cnt:,} (未被修改)')

    # 新的因子类型数
    new_names = [r[0] for r in s.execute(text(
        'SELECT DISTINCT factor_name FROM factor_value WHERE factor_name IN (SELECT DISTINCT factor_name FROM factor_value EXCEPT SELECT DISTINCT factor_name FROM factor_value WHERE factor_name IN (SELECT DISTINCT factor_name FROM factor_value LIMIT 9))'
    )).fetchall()]
    all_names = [r[0] for r in s.execute(text('SELECT DISTINCT factor_name FROM factor_value')).fetchall()]
    print(f'因子类型: {len(all_names)} → {all_names}')
    print('\n[OK] 财务因子补充完成')
