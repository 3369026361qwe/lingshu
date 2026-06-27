"""
简化回测: fusion_score → portfolio_snapshot + risk_logs + var_records。

Strategy: 等权持有 Top 20 选股，40 天调仓一次。
使用 huice.backtest_engine.DBBacktestRunner 封装所有逻辑。
"""
import argparse
import time
from huice.backtest_engine import DBBacktestRunner

# ── 默认参数 ──────────────────────────────────────────
TOP_N = 20
REBAL_FREQ = 40
INITIAL_CAPITAL = 1_000_000.0


def main():
    parser = argparse.ArgumentParser(description='简化回测 — fusion_score 选股回测')
    parser.add_argument('--start', default=None, help='开始日期 YYYYMMDD')
    parser.add_argument('--end', default=None, help='结束日期 YYYYMMDD')
    parser.add_argument('--top-n', type=int, default=TOP_N)
    parser.add_argument('--rebalance', type=int, default=REBAL_FREQ)
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL)
    parser.add_argument('--no-persist', action='store_true', help='不写入数据库')
    args = parser.parse_args()

    t0 = time.time()

    runner = DBBacktestRunner()
    report = runner.run(
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        rebalance_freq=args.rebalance,
        initial_capital=args.capital,
        signal_source='fusion_score',
        persist=not args.no_persist,
    )

    runner.print_summary(report)
    print(f'\n总耗时: {time.time() - t0:.0f}s')
    print('[OK] 简化回测完成')


if __name__ == '__main__':
    main()
