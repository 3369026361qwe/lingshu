"""
从数据库直接计算因子 — daily_bar → factor_value。
覆盖全部 A 股，支持增量（跳过已有日期）。

Usage:
    python scripts/compute_factors_from_db.py              # 全量计算
    python scripts/compute_factors_from_db.py --days 60    # 仅最近 60 天
"""
import sys, os, time, argparse
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
from math import isnan, isinf, sqrt

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

from shujuku.session import SessionContext
from sqlalchemy import text
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--days', type=int, default=0, help='仅计算最近 N 天，0=全量')
args = parser.parse_args()

# ── 因子定义 ──────────────────────────────────────────
# 每类因子: (name, compute_fn)
# compute_fn takes (close_prices: list[float], volumes: list[float], dates: list[str], financials: dict)
# and returns float or None

def safe_decimal(v):
    """Safe string/float to float."""
    if v is None: return None
    try:
        f = float(str(v))
        if isnan(f) or isinf(f): return None
        return f
    except: return None


def compute_ret_1m(prices):
    """1月动量 (约 21 个交易日)"""
    if len(prices) < 22: return None
    return (prices[-1] - prices[-22]) / prices[-22] if prices[-22] != 0 else None

def compute_ret_3m(prices):
    """3月动量"""
    if len(prices) < 64: return None
    return (prices[-1] - prices[-64]) / prices[-64] if prices[-64] != 0 else None

def compute_ret_6m(prices):
    """6月动量"""
    if len(prices) < 126: return None
    return (prices[-1] - prices[-126]) / prices[-126] if prices[-126] != 0 else None

def compute_vol_1m(prices):
    """1月波动率"""
    if len(prices) < 22: return None
    rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-21, 0) if prices[i-1] != 0]
    return float(np.std(rets)) if len(rets) >= 15 else None

def compute_vol_3m(prices):
    """3月波动率"""
    if len(prices) < 64: return None
    rets = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-63, 0) if prices[i-1] != 0]
    return float(np.std(rets)) if len(rets) >= 40 else None

def compute_turnover_1m(volumes, turnovers):
    """1月日均换手率"""
    if len(turnovers) < 22: return None
    valid = [t for t in turnovers[-22:] if t is not None]
    return float(np.mean(valid)) if len(valid) >= 15 else None

def compute_volume_ratio(prices, volumes):
    """量比: 近5日均量 / 近20日均量"""
    if len(volumes) < 20: return None
    v5 = np.mean([v for v in volumes[-5:] if v is not None])
    v20 = np.mean([v for v in volumes[-20:] if v is not None])
    return float(v5 / v20) if v20 and v20 > 0 else None

def compute_rsi(prices):
    """14 日 RSI"""
    if len(prices) < 15: return None
    gains, losses = [], []
    for i in range(-14, 0):
        diff = prices[i] - prices[i-1]
        if diff > 0: gains.append(diff); losses.append(0)
        else: gains.append(0); losses.append(-diff)
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))

def compute_ma_gap(prices):
    """5日偏离度: (Close - MA5) / MA5"""
    if len(prices) < 5: return None
    ma5 = np.mean(prices[-5:])
    return float((prices[-1] - ma5) / ma5) if ma5 != 0 else None

# 因子注册表
FACTOR_CATEGORY = {
    'momentum_1m': 'momentum', 'momentum_3m': 'momentum', 'momentum_6m': 'momentum',
    'historical_vol': 'volatility', 'vol_3m': 'volatility',
    'turnover_momentum': 'sentiment', 'volume_ratio': 'sentiment',
    'rsi_14': 'momentum', 'ma5_gap': 'momentum',
}

FACTOR_DEFS = [
    ('momentum_1m', compute_ret_1m),
    ('momentum_3m', compute_ret_3m),
    ('momentum_6m', compute_ret_6m),
    ('historical_vol', compute_vol_1m),
    ('vol_3m', compute_vol_3m),
    ('turnover_momentum', compute_turnover_1m),
    ('volume_ratio', compute_volume_ratio),
    ('rsi_14', compute_rsi),
    ('ma5_gap', compute_ma_gap),
]

# ── 主流程 ───────────────────────────────────────────

print('=' * 60)
print('从数据库计算因子: daily_bar → factor_value')
print(f'因子数量: {len(FACTOR_DEFS)}')
print()

with SessionContext() as s:
    # 获取所有 stock codes
    codes = [r[0] for r in s.execute(text('SELECT code FROM stock_info WHERE is_active=1')).fetchall()]
    print(f'活跃股票: {len(codes)}')

    # 获取目标日期
    if args.days > 0:
        dates = [r[0] for r in s.execute(text(
            'SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date DESC'
        )).fetchall()[:args.days]]
    else:
        dates = [r[0] for r in s.execute(text(
            'SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date'
        )).fetchall()]

    print(f'目标日期: {len(dates)} ({dates[0]} ~ {dates[-1]})')

    # 检查已有因子数据，跳过已计算的日期
    existing_dates = set(r[0] for r in s.execute(text(
        'SELECT DISTINCT trade_date FROM factor_value'
    )).fetchall())

    new_dates = [d for d in dates if d not in existing_dates]
    print(f'已有数据: {len(existing_dates)} 天 | 需计算: {len(new_dates)} 天')

    if not new_dates:
        print('\n所有日期的因子数据已存在，无需计算。')
        sys.exit(0)

# ── 批量计算 ─────────────────────────────────────────

BATCH_COMMIT = 10  # 每 N 天提交一次
total_factors = 0
t0 = time.time()

for di, trade_date in enumerate(new_dates):
    dt_start = time.time()

    with SessionContext() as s:
        # 获取该日期所有股票的日线数据（近 130 天回溯用于动量因子）
        rows = s.execute(text("""
            SELECT code, close, volume, turnover_rate
            FROM daily_bar
            WHERE trade_date <= :d
            ORDER BY code, trade_date
        """), {"d": str(trade_date)}).fetchall()

    # 组织数据: {code: {'close': [...], 'volume': [...], 'turnover': [...]}}
    stock_data = defaultdict(lambda: {'close': [], 'volume': [], 'turnover': []})
    for code, close, vol, turnover in rows:
        c = safe_decimal(close)
        v = safe_decimal(vol)
        t = safe_decimal(turnover) if turnover else None
        if c is not None:
            stock_data[code]['close'].append(c)
            stock_data[code]['volume'].append(v)
            stock_data[code]['turnover'].append(t)

    # 计算因子
    batch_vals = []
    for code in codes:
        prices = stock_data[code]['close']
        volumes = stock_data[code]['volume']
        turnovers = stock_data[code]['turnover']

        if len(prices) < 7:
            continue

        def _compute(fname, ffunc):
            # 根据因子名决定传入哪些参数
            if fname in ('volume_ratio',):
                return ffunc(prices, volumes)
            elif fname in ('turnover_momentum',):
                return ffunc(volumes, turnovers)
            else:
                return ffunc(prices)

        for fname, ffunc in FACTOR_DEFS:
            try:
                v = _compute(fname, ffunc)
            except Exception:
                continue
            if v is not None and not isnan(v) and not isinf(v) and abs(v) < 1e8:
                batch_vals.append({
                    'c': code, 'd': trade_date, 'fn': fname,
                    'cat': FACTOR_CATEGORY.get(fname, 'value'),
                    'rv': str(round(v, 8)),
                    'zs': str(round(v, 6)),
                })

    # 写入
    if batch_vals:
        with SessionContext() as s:
            for i in range(0, len(batch_vals), 2000):
                s.execute(text(
                    "INSERT INTO factor_value (code, trade_date, category, factor_name, raw_value, z_score, updated_at) "
                    "VALUES (:c, :d, :cat, :fn, :rv, :zs, datetime('now')) "
                    "ON CONFLICT(code, trade_date, factor_name) DO NOTHING"
                ), batch_vals[i:i+2000])
            s.commit()
            total_factors += len(batch_vals)

    elapsed = time.time() - dt_start
    if (di + 1) % 10 == 0 or di == 0:
        total_elapsed = time.time() - t0
        print(f'  [{di+1}/{len(new_dates)}] {trade_date} | {len(batch_vals):>8,} factor values | {elapsed:.1f}s | total: {total_elapsed:.0f}s')

# ── 总结 ─────────────────────────────────────────────

total_time = time.time() - t0
print(f'\n[OK] 因子计算完成')
print(f'  计算日期: {len(new_dates)} 天')
print(f'  因子条目: {total_factors:,}')
print(f'  总耗时: {total_time:.0f}s')
print(f'  下一步: python scripts/run_daily_pipeline.py')
