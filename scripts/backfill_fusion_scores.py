"""
自动补齐 fusion_score — 扫描 factor_value 中缺失融合分数的日期并生成。
每次扩展 factor_value 后运行此脚本即可，无需手动处理。

铁律: INSERT ON CONFLICT DO NOTHING，不覆盖不删除已有数据。

Usage:
    python scripts/backfill_fusion_scores.py                # 补齐所有缺口
    python scripts/backfill_fusion_scores.py --top-n 100    # 每日 Top 100
    python scripts/backfill_fusion_scores.py --dry-run      # 仅检查缺口，不写入
"""
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from shujuku.session import SessionContext
from sqlalchemy import text
from juece.factor_fusion import FactorFusion

parser = argparse.ArgumentParser(description='自动补齐 fusion_score 缺口')
parser.add_argument('--top-n', type=int, default=40, help='每日保留前 N 只股票 (默认 40)')
parser.add_argument('--dry-run', action='store_true', help='仅检查缺口，不写入')
args = parser.parse_args()

print('=' * 60)
print('fusion_score 自动补齐')
print('=' * 60)

with SessionContext() as s:
    # 1. 检查缺口
    fv_dates = set(r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM factor_value ORDER BY trade_date'
    )).fetchall())
    fs_dates = set(r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM fusion_score ORDER BY trade_date'
    )).fetchall())
    missing = sorted(fv_dates - fs_dates)

    fs_before = len(fs_dates)
    print(f'factor_value 日期: {len(fv_dates)}')
    print(f'fusion_score 日期: {fs_before}')
    print(f'缺口: {len(missing)} 天')

    if not missing:
        print('\n无需补齐，退出。')
        sys.exit(0)

    if args.dry_run:
        print(f'\n[dry-run] 将补齐 {len(missing)} 天，不写入。')
        if len(missing) <= 10:
            for d in missing:
                print(f'  {d}')
        sys.exit(0)

    # 2. 批量生成
    print(f'\n补齐 {len(missing)} 天 (Top {args.top_n})...')
    fusion = FactorFusion(min_valid_factors=4)
    total_new = 0
    t0 = time.time()

    for i, trade_date in enumerate(missing):
        rows = s.execute(text(
            f"SELECT code, factor_name, raw_value FROM factor_value WHERE trade_date = '{trade_date}'"
        )).fetchall()
        if not rows:
            continue

        fv = defaultdict(dict)
        for code, fn, rv in rows:
            try: fv[fn][code] = float(str(rv))
            except: pass

        composite = fusion.compute(fv, do_industry_neutralize=False)
        if not composite:
            continue

        ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)[:args.top_n]
        stmt = text(
            "INSERT INTO fusion_score (trade_date, code, composite_score, rank, signal) "
            "VALUES (:d, :c, :s, :r, 'HOLD') "
            "ON CONFLICT(trade_date, code) DO NOTHING"
        )
        batch = [{'d': trade_date, 'c': c, 's': str(round(sc, 6)), 'r': ri}
                 for ri, (c, sc) in enumerate(ranked, 1)]
        if batch:
            s.execute(stmt, batch)
            s.commit()
            total_new += len(batch)

        if (i + 1) % 30 == 0:
            print(f'  进度: {i+1}/{len(missing)} ({time.time()-t0:.0f}s)')

    # 3. 验证
    fs_after = s.execute(text(
        'SELECT COUNT(DISTINCT trade_date) FROM fusion_score'
    )).scalar()
    fs_rows = s.execute(text('SELECT COUNT(*) FROM fusion_score')).scalar()

    print(f'\n  补齐完成:')
    print(f'    fusion_score 日期: {fs_before} -> {fs_after} (+{fs_after-fs_before})')
    print(f'    fusion_score 行数: {fs_rows:,} (+{total_new})')
    print(f'    耗时: {time.time()-t0:.0f}s')

    # 4. 安全验证
    fv_cnt = s.execute(text('SELECT COUNT(*) FROM factor_value')).scalar()
    print(f'\n  factor_value: {fv_cnt:,} (未修改)')
    print('[OK] backfill 完成')
