"""
最终回测验证 — 策略对比: Factor-only | GNN-only | Fusion Static | Fusion IC-Dynamic。

参数: TopN=40, Freq=40d, W=50/30/20 (网格搜索最优)
模型: data/gnn_model.pt
"""
import argparse
import time
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import torch
from dotenv import load_dotenv

load_dotenv(Path('E:/28721/lingshu/.env'))

from tushenjing.graph_inference import load_gnn_checkpoint, run_gnn_inference

# ── 默认参数 ──────────────────────────────────────────
TOP_N = 40
FREQ = 40
CAPITAL = 1_000_000.0
BASE = Path('E:/28721/lingshu/data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def norm(scores_dict):
    """Min-max 归一化。"""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    v_min, v_max = min(vals), max(vals)
    if v_max > v_min:
        return {c: (v - v_min) / (v_max - v_min) for c, v in scores_dict.items()}
    return {c: 0.5 for c in scores_dict}


def backtest_scores(scores_by_date, label, close_map, all_dates_ordered,
                    top_n=TOP_N, freq=FREQ, capital=CAPITAL):
    """对给定信号运行回测，返回绩效指标。"""
    dates = sorted(scores_by_date.keys())
    rebals = dates[::freq]
    rebals = [d for d in rebals if d >= dates[min(6, len(dates) // 10)]]
    if len(rebals) < 5:
        return None

    cash = capital
    holdings = {}
    pv = []
    rets = []

    for ri, rd in enumerate(rebals):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:top_n]
        picks = set(c for c, _ in ranked)
        nd = rebals[ri + 1] if ri + 1 < len(rebals) else dates[-1]

        for c in list(holdings.keys()):
            px = close_map.get(c, {}).get(rd)
            if px and px > 0:
                cash += holdings[c]['shares'] * px * 0.999
            del holdings[c]

        pp = {c: close_map.get(c, {}).get(rd) for c in picks}
        pp = {c: p for c, p in pp.items() if p and p > 0}
        if len(pp) < max(5, top_n // 3):
            continue

        ps = cash / len(pp)
        for c, p in pp.items():
            sh = int(ps / (p * 1.001) / 100) * 100
            if sh > 0:
                cash -= sh * p * 1.001
                holdings[c] = {'shares': sh}

        for d in [x for x in all_dates_ordered if rd <= x < nd]:
            mv = cash + sum(
                holdings[c]['shares'] * close_map.get(c, {}).get(d, 0)
                for c in holdings if close_map.get(c, {}).get(d)
            )
            pv.append(mv)
            if len(pv) >= 2 and pv[-2] > 0:
                rets.append((pv[-1] - pv[-2]) / pv[-2])

    if not rets:
        return None

    fv = pv[-1]
    tr = (fv - capital) / capital * 100
    ny = len(rets) / 252
    ar = ((fv / capital) ** (1 / ny) - 1) * 100 if ny > 0 else 0
    mean(rets)
    sg = stdev(rets) if len(rets) > 1 else 0.01
    av = sg * sqrt(252) * 100
    sh = (ar - 2.5) / av if av > 0 else 0
    pk = capital
    dd = 0
    for m in pv:
        if m > pk:
            pk = m
        d = (m - pk) / pk * 100
        if d < dd:
            dd = d
    wr = sum(1 for r in rets if r > 0) / len(rets) * 100
    return {
        'label': label, 'total_ret': tr, 'ann_ret': ar, 'ann_vol': av,
        'sharpe': sh, 'max_dd': dd, 'win_rate': wr, 'n_rebals': len(rebals),
    }


def main():
    parser = argparse.ArgumentParser(description='最终回测验证 — 四策略对比')
    parser.add_argument('--top-n', type=int, default=TOP_N)
    parser.add_argument('--freq', type=int, default=FREQ)
    parser.add_argument('--capital', type=float, default=CAPITAL)
    parser.add_argument('--model', default=str(BASE / 'gnn_model.pt'), help='GNN checkpoint 路径')
    args = parser.parse_args()

    print('=' * 60)
    print('  灵枢量化 — 最终回测验证')
    print(f'  参数: TopN={args.top_n} | Freq={args.freq}d | {DEVICE}')
    print('=' * 60)

    from sqlalchemy import text

    from shujuku.session import SessionContext

    # 1 ── 加载数据 ──
    print('\n[1/4] Loading data...')
    t0 = time.time()
    with SessionContext() as s:
        fv_rows = s.execute(text(
            'SELECT code, trade_date, factor_name, raw_value '
            'FROM factor_value ORDER BY trade_date, code, factor_name'
        )).fetchall()
        price_rows = s.execute(text(
            'SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date'
        )).fetchall()
        fusion_rows = s.execute(text(
            'SELECT trade_date, code, composite_score FROM fusion_score ORDER BY trade_date, code'
        )).fetchall()
    print(f'  {len(fv_rows):,} factors | {len(price_rows):,} prices | {len(fusion_rows):,} fusion ({time.time() - t0:.1f}s)')

    close_map = defaultdict(dict)
    for code, td, cl in price_rows:
        close_map[code][str(td)] = float(cl)
    all_dates = sorted(set(str(r[1]) for r in price_rows))

    factor_scores = defaultdict(dict)
    for td, code, sc in fusion_rows:
        factor_scores[str(td)][code] = float(sc)

    factor_dates = sorted(factor_scores.keys())
    train_cut = int(len(factor_dates) * 0.6)
    test_dates = factor_dates[train_cut:]
    print(f'  Test dates: {len(test_dates)} (last 40%)')

    # 2 ── GNN 推理 ──
    print('\n[2/4] GNN inference...')
    t0 = time.time()
    ckpt = load_gnn_checkpoint(args.model, str(DEVICE))
    if ckpt and ckpt['model']:
        factor_by_date = defaultdict(lambda: defaultdict(dict))
        for code, td, fn, rv in fv_rows:
            try:
                factor_by_date[str(td)][code][fn] = float(str(rv))
            except Exception:
                pass
        gnn_scores = run_gnn_inference(ckpt, factor_by_date, str(DEVICE))
        print(f'  {ckpt["model_type"]} | {len(gnn_scores)} dates ({time.time() - t0:.1f}s)')
    else:
        print('  No GNN model found, skipping')
        gnn_scores = {}

    # 3 ── 策略对比 ──
    print('\n[3/4] Strategy comparison...')
    # Filter to test dates
    factor_test = {td: factor_scores[td] for td in test_dates if td in factor_scores}
    gnn_test = {td: gnn_scores[td] for td in test_dates if td in gnn_scores}

    # Fusion Static (50/30/20)
    fusion_static = defaultdict(dict)
    for td in test_dates:
        fn = norm(factor_scores.get(td, {}))
        gn = norm(gnn_scores.get(td, {}))
        for c in set(fn) | set(gn):
            fusion_static[td][c] = fn.get(c, 0.5) * 0.50 + gn.get(c, 0.5) * 0.30 + 0.5 * 0.20

    # Fusion IC-Dynamic (20/45/35 — from IC-based weights)
    fusion_dynamic = defaultdict(dict)
    for td in test_dates:
        fn = norm(factor_scores.get(td, {}))
        gn = norm(gnn_scores.get(td, {}))
        for c in set(fn) | set(gn):
            fusion_dynamic[td][c] = fn.get(c, 0.5) * 0.20 + gn.get(c, 0.5) * 0.45 + 0.5 * 0.35

    results = []
    for scores, lbl in [
        (factor_test, 'Factor-only'),
        (gnn_test, 'GNN-only'),
        (fusion_static, 'Fusion Static'),
        (fusion_dynamic, 'Fusion IC-Dynamic'),
    ]:
        r = backtest_scores(scores, lbl, close_map, all_dates,
                           args.top_n, args.freq, args.capital)
        if r:
            results.append(r)
            print(f'  {lbl:20s}: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

    # 4 ── 报告 ──
    print(f'\n{"=" * 60}')
    print('  最终回测报告')
    print(f'{"=" * 60}')
    print(f'  {"Strategy":20s} {"Return":>8s} {"Sharpe":>8s} {"MaxDD":>8s} {"Vol":>8s} {"Win%":>7s}')
    print(f'  {"-" * 55}')
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f'  {r["label"]:20s} {r["total_ret"]:>+7.1f}% {r["sharpe"]:>8.3f} {r["max_dd"]:>7.1f}% {r["ann_vol"]:>7.1f}% {r["win_rate"]:>6.1f}%')

    if results:
        best = max(results, key=lambda x: x['sharpe'])
        print(f'\n  Best: {best["label"]} (Sharpe={best["sharpe"]:.3f})')

    print(f'{"=" * 60}')
    print('  灵枢量化 — 最终回测完成')


if __name__ == '__main__':
    main()
