"""
全周期回测: 1132天完整绩效 + 滚动夏普 + 年度收益 + 风控回测。

Factor-only 全周期 / GNN+Fusion 仅测试期（避免前视偏差）。
使用 tushenjing.graph_inference 进行 GNN 推理。
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from math import sqrt
from statistics import mean, stdev

import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))

from huice.backtest_engine import DBBacktestRunner
from tushenjing.graph_inference import load_gnn_checkpoint, run_gnn_inference

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_N = 40
FREQ = 40
CAPITAL = 1_000_000.0


def norm(scores_dict):
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    v_min, v_max = min(vals), max(vals)
    if v_max > v_min:
        return {c: (v - v_min) / (v_max - v_min) for c, v in scores_dict.items()}
    return {c: 0.5 for c in scores_dict}


def backtest_with_metrics(scores_by_date, label, close_map, all_dates_ordered,
                          top_n=TOP_N, freq=FREQ, capital=CAPITAL):
    """带完整指标的回测：总收益/年化/夏普/最大回撤/胜率/滚动夏普/年度收益。"""
    dates = sorted(scores_by_date.keys())
    rebals = dates[::freq]
    rebals = [d for d in rebals if d >= dates[min(6, len(dates) // 10)]]
    if len(rebals) < 5:
        return None

    cash = capital
    holdings = {}
    pv = []
    rets = []
    rolling_sharpes = []
    yr_rets = defaultdict(list)

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
            pv.append((d, mv))
            if len(pv) >= 2 and pv[-2][1] > 0:
                ret = (mv - pv[-2][1]) / pv[-2][1]
                rets.append(ret)
                yr_rets[d[:4]].append(ret)

        if len(rets) >= 60:
            r60 = rets[-60:]
            mu60 = mean(r60)
            sg60 = stdev(r60) if len(r60) > 1 else 0.01
            rolling_sharpes.append((mu60 / sg60) * sqrt(252))

    if not rets:
        return None

    fv_mv = pv[-1][1]
    tr = (fv_mv - capital) / capital * 100
    ny = len(rets) / 252
    ar = ((fv_mv / capital) ** (1 / ny) - 1) * 100 if ny > 0 else 0
    mu = mean(rets)
    sg = stdev(rets) if len(rets) > 1 else 0.01
    av = sg * sqrt(252) * 100
    sh = (ar - 2.5) / av if av > 0 else 0
    pk = capital
    dd = 0
    for _, mv in pv:
        if mv > pk:
            pk = mv
        d = (mv - pk) / pk * 100
        if d < dd:
            dd = d
    wr = sum(1 for r in rets if r > 0) / len(rets) * 100
    annual = {yr: (sum(r) * 100, len(r)) for yr, r in yr_rets.items() if len(r) > 50}

    return {
        'label': label, 'total_ret': tr, 'ann_ret': ar, 'ann_vol': av,
        'sharpe': sh, 'max_dd': dd, 'win_rate': wr, 'n_rebals': len(rebals),
        'n_days': len(rets), 'rolling_sharpes': rolling_sharpes,
        'annual': annual, 'final_mv': fv_mv,
    }


def backtest_with_risk(scores_by_date, label, close_map, all_dates_ordered,
                       top_n=TOP_N, freq=FREQ, capital=CAPITAL):
    """带分级风控的回测：L0-L3 回撤缩放。"""
    dates = sorted(scores_by_date.keys())
    rebals = dates[::freq]
    rebals = [d for d in rebals if d >= dates[min(6, len(dates) // 10)]]
    if len(rebals) < 5:
        return None, []

    cash = capital
    holdings = {}
    pv = []
    rets = []
    risk_events = []
    peak = capital
    dd_count = 0

    for ri, rd in enumerate(rebals):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:top_n]
        picks = set(c for c, _ in ranked)
        nd = rebals[ri + 1] if ri + 1 < len(rebals) else dates[-1]

        equity = cash + sum(
            holdings[c]['shares'] * close_map.get(c, {}).get(rd, 0)
            for c in holdings if close_map.get(c, {}).get(rd, 0)
        )
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > 0.20:
            scale = 0.25
            risk_level = 'CRITICAL'
        elif dd > 0.15:
            scale = 0.50
            risk_level = 'HIGH'
        elif dd > 0.10:
            scale = 0.75
            risk_level = 'ELEVATED'
        else:
            scale = 1.0
            risk_level = 'LOW'

        if scale < 1.0:
            dd_count += 1
            risk_events.append({'date': rd, 'dd': round(dd * 100, 1), 'scale': scale, 'level': risk_level})

        for c in list(holdings.keys()):
            px = close_map.get(c, {}).get(rd)
            if px and px > 0:
                cash += holdings[c]['shares'] * px * 0.999
            del holdings[c]

        pp = {c: close_map.get(c, {}).get(rd) for c in picks}
        pp = {c: p for c, p in pp.items() if p and p > 0}
        if len(pp) < max(5, top_n // 3):
            continue

        investable = cash * scale
        ps = investable / len(pp)
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
        return None, []

    fv_mv = pv[-1]
    tr = (fv_mv - capital) / capital * 100
    ny = len(rets) / 252
    ar = ((fv_mv / capital) ** (1 / ny) - 1) * 100 if ny > 0 else 0
    mu = mean(rets)
    sg = stdev(rets) if len(rets) > 1 else 0.01
    av = sg * sqrt(252) * 100
    sh = (ar - 2.5) / av if av > 0 else 0
    pk = capital
    dd_max = 0
    for mv in pv:
        if mv > pk:
            pk = mv
        d = (mv - pk) / pk * 100
        if d < dd_max:
            dd_max = d

    return {
        'label': label, 'total_ret': tr, 'ann_ret': ar, 'ann_vol': av,
        'sharpe': sh, 'max_dd': dd_max,
        'win_rate': sum(1 for r in rets if r > 0) / len(rets) * 100,
        'n_days': len(rets), 'n_rebals': len(rebals), 'dd_triggers': dd_count,
    }, risk_events


def main():
    parser = argparse.ArgumentParser(description='全周期回测 — GNN+Fusion+风控')
    parser.add_argument('--top-n', type=int, default=TOP_N)
    parser.add_argument('--freq', type=int, default=FREQ)
    parser.add_argument('--capital', type=float, default=CAPITAL)
    parser.add_argument('--model', default=str(BASE / 'gnn_model.pt'), help='GNN checkpoint 路径')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  全周期回测 | TopN={args.top_n} Freq={args.freq}d | {DEVICE}')
    print('=' * 60)

    from shujuku.session import SessionContext
    from sqlalchemy import text

    # 1 ── 加载数据 ──
    print('\n[1/4] Loading...')
    t0 = time.time()
    with SessionContext() as s:
        fv_rows = s.execute(text(
            'SELECT code, trade_date, factor_name, raw_value '
            'FROM factor_value ORDER BY trade_date, code, factor_name'
        )).fetchall()
        pr_rows = s.execute(text(
            'SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date'
        )).fetchall()
        fs_rows = s.execute(text(
            'SELECT trade_date, code, composite_score FROM fusion_score ORDER BY trade_date, code'
        )).fetchall()
    print(f'  {len(fv_rows):,} factors | {len(pr_rows):,} prices | {len(fs_rows):,} fusion ({time.time() - t0:.1f}s)')

    close_map = defaultdict(dict)
    for c, td, cl in pr_rows:
        close_map[c][str(td)] = float(cl)
    all_dates = sorted(set(str(r[1]) for r in pr_rows))
    date_index = {d: i for i, d in enumerate(all_dates)}

    factor_scores = defaultdict(dict)
    for td, c, sc in fs_rows:
        factor_scores[str(td)][c] = float(sc)

    # 2 ── GNN 推理 ──
    print('[2/4] Loading GNN...')
    t0 = time.time()
    ckpt = load_gnn_checkpoint(args.model, str(DEVICE))
    gnn_scores = {}
    if ckpt and ckpt['model']:
        factor_by_date = defaultdict(lambda: defaultdict(dict))
        for c, td, fn, rv in fv_rows:
            try:
                factor_by_date[str(td)][c][fn] = float(str(rv))
            except Exception:
                pass
        gnn_scores = run_gnn_inference(ckpt, factor_by_date, str(DEVICE))
        print(f'  {ckpt["model_type"]} | {len(gnn_scores)} dates ({time.time() - t0:.1f}s)')
    else:
        print(f'  No GNN model')

    # 3 ── 策略回测 ──
    print('[3/4] Running backtests...')
    results = []

    # Factor-only: 全周期
    r = backtest_with_metrics(factor_scores, 'Factor (全周期)', close_map, all_dates, args.top_n, args.freq, args.capital)
    if r:
        results.append(r)
        print(f'  Factor全周期: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

    # GNN + Fusion: 仅测试期
    all_common = sorted(set(factor_scores.keys()) & set(gnn_scores.keys()))
    train_cut = int(len(all_common) * 0.6)
    test_dates = all_common[train_cut:]
    gnn_test = {td: gnn_scores[td] for td in test_dates if td in gnn_scores}
    factor_test = {td: factor_scores[td] for td in test_dates if td in factor_scores}

    # Agent proxy: 市场动量信号
    agent_signals = {}
    for di, td in enumerate(test_dates):
        mdi = date_index.get(td, -1)
        if mdi >= 20:
            ret_20d = 0
            sample_codes = list(close_map.keys())[:100]
            for i in range(mdi - 20, mdi):
                prev_prices = [close_map.get(c, {}).get(all_dates[i], 0) for c in sample_codes]
                curr_prices = [close_map.get(c, {}).get(all_dates[i + 1], 0) for c in sample_codes]
                if prev_prices and curr_prices:
                    ret_20d += (mean(curr_prices) - mean(prev_prices)) / mean(prev_prices)
            agent_signals[td] = max(0.1, min(0.9, 0.5 + ret_20d * 2))
        else:
            agent_signals[td] = 0.5

    # Fusion (20/45/35 with Agent)
    fusion_dynamic = defaultdict(dict)
    fusion_static = defaultdict(dict)
    for td in test_dates:
        fn = norm(factor_scores.get(td, {}))
        gn = norm(gnn_scores.get(td, {}))
        ag = agent_signals.get(td, 0.5)
        for c in set(fn) | set(gn):
            fusion_dynamic[td][c] = fn.get(c, 0.5) * 0.20 + gn.get(c, 0.5) * 0.45 + ag * 0.35
            fusion_static[td][c] = fn.get(c, 0.5) * 0.50 + gn.get(c, 0.5) * 0.30 + 0.5 * 0.20

    for scores, lbl in [
        (factor_test, 'Factor (测试期)'),
        (gnn_test, 'GNN GAT'),
        (fusion_dynamic, 'Fusion(20/45/35)+Agent'),
        (fusion_static, 'Fusion旧(50/30/20)'),
    ]:
        r = backtest_with_metrics(scores, lbl, close_map, all_dates, args.top_n, args.freq, args.capital)
        if r:
            results.append(r)
            print(f'  {lbl:25s}: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

    print(f'  Agent avg signal: {mean(agent_signals.values()):.3f}')

    # 4 ── 风控回测 ──
    print(f'\n[4/4] Risk-managed backtest...')
    r_risk, risk_events = backtest_with_risk(
        factor_scores, 'Factor + 风控', close_map, all_dates, args.top_n, args.freq, args.capital
    )
    if r_risk:
        results.append(r_risk)
        print(f'  Factor+风控: Ret={r_risk["total_ret"]:+.1f}% Sharpe={r_risk["sharpe"]:.3f} DD={r_risk["max_dd"]:.1f}% (触发{r_risk["dd_triggers"]}次)')

    if risk_events:
        with SessionContext() as s:
            stmt = text(
                "INSERT INTO risk_logs (timestamp, level, category, message, detail, updated_at) "
                "VALUES (datetime('now'), :l, 'DRAWDOWN', '回撤触发仓位缩减', :d, datetime('now'))"
            )
            for ev in risk_events[:50]:
                try:
                    s.execute(stmt, {'l': ev['level'], 'd': f"date={ev['date']} dd={ev['dd']}% scale={ev['scale']:.0%}"})
                except Exception:
                    pass
            s.commit()

    # ── 报告 ──
    print(f'\n{"=" * 60}')
    print(f'  全周期回测报告')
    print(f'{"=" * 60}')
    print(f'  {"Strategy":25s} {"Return":>8s} {"Sharpe":>8s} {"MaxDD":>8s} {"Vol":>8s} {"Win%":>7s} {"Days":>6s}')
    print(f'  {"-" * 65}')
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        av = r.get('ann_vol', 0)
        print(f'  {r["label"]:25s} {r["total_ret"]:>+7.1f}% {r["sharpe"]:>8.3f} {r["max_dd"]:>7.1f}% {av:>7.1f}% {r["win_rate"]:>6.1f}% {r.get("n_days", 0):>6d}')

    # Annual
    r_full = [r for r in results if '全周期' in r['label'] and '风控' not in r['label']]
    if r_full and r_full[0].get('annual'):
        print(f'\n  ── 年度收益 ──')
        for yr in sorted(r_full[0]['annual'].keys()):
            ret, n = r_full[0]['annual'][yr]
            bar = ('+' if ret > 0 else '') + '█' * int(abs(ret) / 5)
            print(f'    {yr}: {ret:+.1f}% ({n}天) {bar}')

    # Rolling Sharpe
    if r_full and r_full[0].get('rolling_sharpes'):
        rs = r_full[0]['rolling_sharpes']
        if rs:
            print(f'\n  ── 滚动夏普(60期) ──')
            print(f'    Mean: {mean(rs):.2f} | Max: {max(rs):.2f} | Min: {min(rs):.2f} | Final: {rs[-1]:.2f}')

    # Save
    json.dump(
        [{k: v for k, v in r.items() if k not in ('rolling_sharpes', 'annual')} for r in results],
        open(BASE / 'full_period_report.json', 'w'), indent=2, default=str
    )
    print(f'\n  Report saved')

    print(f'{"=" * 60}')
    print('  全周期回测 + 风控完成')


if __name__ == '__main__':
    main()
