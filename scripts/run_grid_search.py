"""
网格搜索：TopN × 调仓频率 二维优化 (多进程并行)。

使用 huice.backtest_engine.DBBacktestRunner + ProcessPoolExecutor。
"""
import argparse
import time
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from shujuku.session import SessionContext
from sqlalchemy import text
from huice.backtest_engine import DBBacktestRunner

TOP_N_VALUES = [10, 20, 30, 40, 50]
FREQ_VALUES = [5, 10, 20, 40, 60]
CAPITAL = 1_000_000.0
MAX_WORKERS = min(os.cpu_count() or 4, len(TOP_N_VALUES) * len(FREQ_VALUES))


def _run_one(params):
    """单个参数组合回测 (worker 函数, 每个进程独立创建 runner)。"""
    top_n, freq, start_date, end_date = params
    runner = DBBacktestRunner()
    report = runner.run(
        start_date=start_date, end_date=end_date,
        top_n=top_n, rebalance_freq=freq,
        initial_capital=CAPITAL,
        signal_source='fusion_score',
        persist=False,
    )
    return {
        'top_n': top_n, 'freq': freq,
        'sharpe': report['sharpe'],
        'total_ret': report['total_return_pct'],
        'max_dd': report['max_drawdown_pct'],
    }


def main():
    parser = argparse.ArgumentParser(description='网格搜索 — 参数优化')
    parser.add_argument('--start', default=None, help='开始日期 YYYYMMDD')
    parser.add_argument('--end', default=None, help='结束日期 YYYYMMDD')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'并行进程数 (默认: {MAX_WORKERS})')
    args = parser.parse_args()

    print('=' * 65)
    print(f'  网格搜索 — Parameter Optimization ({args.workers} workers)')
    print('=' * 65)

    # 1 ── 加载数据 ──
    print('\n[1/3] Loading data...')
    t0 = time.time()
    with SessionContext() as s:
        fusion_rows = s.execute(text(
            'SELECT trade_date, code, composite_score FROM fusion_score ORDER BY trade_date, code'
        )).fetchall()
        price_rows = s.execute(text(
            'SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date'
        )).fetchall()

    factor_scores = defaultdict(dict)
    for td, code, score in fusion_rows:
        factor_scores[str(td)][code] = float(score)

    common_dates = sorted(set(factor_scores.keys()))
    test_dates = common_dates[len(common_dates) // 2:]

    print(f'  Factor: {len(fusion_rows):,} scores | Test: {len(test_dates)} dates ({time.time() - t0:.1f}s)')

    # 2 ── 网格搜索 (多进程并行) ──
    print(f'\n[2/3] Grid Search: TopN × Frequency ({len(TOP_N_VALUES)}×{len(FREQ_VALUES)}={len(TOP_N_VALUES)*len(FREQ_VALUES)} combos)...')
    t0 = time.time()

    # 构建参数网格
    param_grid = [
        (top_n, freq, args.start, args.end)
        for top_n in TOP_N_VALUES
        for freq in FREQ_VALUES
    ]

    grid_results = []
    best_sharpe = -999
    best_params = None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, p): p for p in param_grid}
        for future in as_completed(futures):
            r = future.result()
            grid_results.append(r)
            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_params = (r['top_n'], r['freq'])
            print(f'  TopN={r["top_n"]:2d} Freq={r["freq"]:2d}d  Sharpe={r["sharpe"]:.3f}  Ret={r["total_ret"]:+.1f}%  DD={r["max_dd"]:.1f}%')

    elapsed = time.time() - t0
    print(f'  Grid done: {len(grid_results)} combinations ({elapsed:.1f}s)')
    print(f'  Best: TopN={best_params[0]}, Freq={best_params[1]}d, Sharpe={best_sharpe:.3f}')

    # 3 ── 报告 ──
    print(f'\n{"=" * 65}')
    print(f'  网格搜索 — Final Report')
    print(f'{"=" * 65}')
    print(f'  {"TopN":>5s} {"Freq":>5s} {"Sharpe":>8s} {"Return":>8s} {"MaxDD":>8s}')
    top = sorted(grid_results, key=lambda x: x['sharpe'], reverse=True)[:10]
    for r in top:
        print(f'  {r["top_n"]:>5d} {r["freq"]:>5d} {r["sharpe"]:>8.3f} {r["total_ret"]:>+7.1f}% {r["max_dd"]:>7.1f}%')

    print(f'\n  BEST CONFIG: TopN={best_params[0]} | Freq={best_params[1]}d | Sharpe={best_sharpe:.3f}')
    print(f'{"=" * 65}')


if __name__ == '__main__':
    main()
