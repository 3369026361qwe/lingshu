"""
AKShare 历史数据延长 — 从 2024-01-01 起补全日线。
ON CONFLICT DO NOTHING，保护已有数据。

Usage:
    python scripts/extend_history_akshare.py              # 全量
    python scripts/extend_history_akshare.py --sample 100 # 仅100只测试
"""
import argparse
import sys
import time
from datetime import date

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import akshare as ak
from sqlalchemy import text

from shujuku.session import SessionContext

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=int, default=0, help='仅下载前N只股票（测试用）')
args = parser.parse_args()

START_DATE = '20240101'
END_DATE = date.today().strftime('%Y%m%d')

print('=' * 60)
print(f'AKShare 历史数据延长: {START_DATE} ~ {END_DATE}')
print('=' * 60)

with SessionContext() as s:
    # 获取已有股票列表
    codes = [r[0] for r in s.execute(text(
        'SELECT code FROM stock_info WHERE is_active=1 ORDER BY code'
    )).fetchall()]

    if args.sample > 0:
        codes = codes[:args.sample]
        print(f'[测试模式] {len(codes)} 只')

    # 获取已有日期的范围
    earliest = s.execute(text('SELECT MIN(trade_date) FROM daily_bar')).scalar()
    latest = s.execute(text('SELECT MAX(trade_date) FROM daily_bar')).scalar()
    before = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
    print(f'已有 daily_bar: {before:,} 行 ({earliest} ~ {latest})')
    print(f'待下载: {len(codes)} 只\n')

    new_bars = 0
    t0 = time.time()

    for i, code in enumerate(codes):
        try:
            # AKShare: stock_zh_a_hist
            # symbol格式: '000001' (不带后缀)
            symbol = code.split('.')[0]
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=START_DATE,
                end_date=END_DATE,
                adjust='qfq'  # 前复权
            )

            if df is None or len(df) == 0:
                continue

            # 转换列名: AKShare返回中文列名
            # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率
            batch = []
            for _, row in df.iterrows():
                try:
                    td = str(row.get('日期', ''))[:10].replace('-', '')
                    if not td or len(td) != 8:
                        continue

                    batch.append({
                        'c': code,
                        'd': td,
                        'o': str(row.get('开盘', '0') or '0'),
                        'h': str(row.get('最高', '0') or '0'),
                        'l': str(row.get('最低', '0') or '0'),
                        'cl': str(row.get('收盘', '0') or '0'),
                        'v': str(row.get('成交量', '0') or '0'),
                        'a': str(row.get('成交额', '0') or '0'),
                        't': str(row.get('换手率', '0') or '0'),
                    })
                except Exception:
                    continue

            if batch:
                stmt = text(
                    "INSERT INTO daily_bar (code, trade_date, open, high, low, close, volume, amount, turnover_rate, is_st, updated_at) "
                    "VALUES (:c, :d, :o, :h, :l, :cl, :v, :a, :t, 0, datetime('now')) "
                    "ON CONFLICT(code, trade_date) DO NOTHING"
                )
                for j in range(0, len(batch), 2000):
                    s.execute(stmt, batch[j:j+2000])
                s.commit()
                new_bars += len(batch)

            # 频率控制
            time.sleep(0.15)

        except Exception as e:
            if '每分钟' in str(e) or 'frequency' in str(e).lower():
                print('  频率限制，等待 60s...')
                time.sleep(60)
                continue
            continue

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            after_now = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
            print(f'  进度: {i+1}/{len(codes)} | DB: {after_now:,} | +{after_now-before:,} | {elapsed:.0f}s')

    after = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
    new_earliest = s.execute(text('SELECT MIN(trade_date) FROM daily_bar')).scalar()

    print(f'\n  完成: {before:,} -> {after:,} (+{after-before:,})')
    print(f'  日期范围: {new_earliest} ~ {latest}')
    print(f'  耗时: {time.time()-t0:.0f}s')
    print('[OK] AKShare 历史延长完成')
