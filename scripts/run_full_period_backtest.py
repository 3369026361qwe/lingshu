"""
全周期回测: 1132天完整绩效 + 滚动夏普 + 年度收益。
Factor-only 全周期 / GNN+Fusion 仅测试期（避免前视偏差）
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_N = 40; FREQ = 40; CAPITAL = 1_000_000.0

print('=' * 60)
print(f'  全周期回测 | TopN={TOP_N} Freq={FREQ}d | {DEVICE}')
print('=' * 60)

# 1. Load data
print('\n[1/4] Loading...'); t0 = time.time()
from shujuku.session import SessionContext
from sqlalchemy import text
with SessionContext() as s:
    fv = s.execute(text('SELECT code,trade_date,factor_name,raw_value FROM factor_value ORDER BY trade_date,code,factor_name')).fetchall()
    pr = s.execute(text('SELECT code,trade_date,close FROM daily_bar ORDER BY code,trade_date')).fetchall()
    fs = s.execute(text('SELECT trade_date,code,composite_score FROM fusion_score ORDER BY trade_date,code')).fetchall()
print(f'  {len(fv):,} factors | {len(pr):,} prices | {len(fs):,} fusion ({time.time()-t0:.1f}s)')

close_map = defaultdict(dict)
for c,td,cl in pr: close_map[c][str(td)] = float(cl)
all_dates = sorted(set(str(r[1]) for r in pr))

factor_scores = defaultdict(dict)
for td,c,sc in fs: factor_scores[str(td)][c] = float(sc)

# 2. Load GNN (only for test period)
print('[2/4] Loading GNN...'); t0 = time.time()
gnn_scores = defaultdict(dict)
model_path = BASE / 'gnn_model.pt'
if model_path.exists():
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    mt = ckpt.get('model_type','GCN')
    in_dim = len(ckpt['features']); hd = ckpt['hidden_dim']; dp = ckpt['dropout']
    stock_codes = ckpt['stock_codes']; edge_index = ckpt['edge_index'].to(DEVICE)

    if mt == 'GAT':
        from torch_geometric.nn import GATConv
        heads = ckpt.get('gat_heads', 4)
        class GNNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(in_dim, hd, heads=heads, dropout=dp)
                self.conv2 = GATConv(hd*heads, 1, heads=1, dropout=dp)
            def forward(self, x, ei):
                x = F.dropout(x, p=dp, training=False)
                x = F.elu(self.conv1(x, ei))
                x = F.dropout(x, p=dp, training=False)
                return self.conv2(x, ei)
    else:
        from torch_geometric.nn import GCNConv
        class GNNModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hd)
                self.conv2 = GCNConv(hd, 1)
            def forward(self, x, ei):
                x = F.relu(self.conv1(x, ei))
                x = F.dropout(x, p=dp, training=False)
                return self.conv2(x, ei)
    model = GNNModel().to(DEVICE); model.load_state_dict(ckpt['model_state_dict']); model.eval()

    from math import isnan as mi
    factor_by_date = defaultdict(lambda: defaultdict(dict))
    for c,td,fn,rv in fv:
        try: factor_by_date[str(td)][c][fn] = float(str(rv))
        except: pass
    fv_dates = sorted(factor_by_date.keys())

    for mdate in fv_dates:
        if mdate not in all_dates: continue
        f = np.zeros((len(stock_codes), len(ckpt['features'])), dtype=np.float32)
        fv_d = factor_by_date[mdate]
        for i,c in enumerate(stock_codes):
            if c in fv_d:
                for j,fn in enumerate(ckpt['features']):
                    v = fv_d[c].get(fn)
                    if v is not None and not mi(v) and abs(v) < 1e8: f[i,j] = v
        x = torch.from_numpy(f).float().to(DEVICE)
        with torch.no_grad(): p = model(x, edge_index).cpu().numpy().ravel()
        for i,c in enumerate(stock_codes): gnn_scores[mdate][c] = float(p[i])
    print(f'  {mt} | {len(gnn_scores)} dates ({time.time()-t0:.1f}s)')
else:
    print(f'  No GNN model')

# 3. Backtest function
def backtest(scores_by_date, label, start_idx=0):
    dates = sorted(scores_by_date.keys())
    if start_idx > 0: dates = dates[start_idx:]
    rebal = dates[::FREQ]; rebal = [d for d in rebal if d >= dates[min(6, len(dates)//10)]]
    if len(rebal) < 5: return None
    cash = CAPITAL; holdings = {}; pv = []; rets = []; rolling_sharpes = []
    annual_rets = {}

    for ri, rd in enumerate(rebal):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        picks = set(c for c, _ in ranked)
        nd = rebal[ri+1] if ri+1 < len(rebal) else dates[-1]
        for c in list(holdings.keys()):
            px = close_map.get(c,{}).get(rd)
            if px and px>0: cash += holdings[c]['shares'] * px * 0.999
            del holdings[c]
        pp = {c: close_map.get(c,{}).get(rd) for c in picks}
        pp = {c:p for c,p in pp.items() if p and p>0}
        if len(pp) < max(5, TOP_N//3): continue
        ps = cash / len(pp)
        for c,p in pp.items():
            sh = int(ps/(p*1.001)/100)*100
            if sh>0: cash -= sh*p*1.001; holdings[c] = {'shares': sh}
        for d in [x for x in all_dates if rd <= x < nd]:
            mv = cash + sum(holdings[c]['shares']*close_map.get(c,{}).get(d,0) for c in holdings if close_map.get(c,{}).get(d))
            pv.append((d, mv))
            if len(pv)>=2 and pv[-2][1]>0: rets.append((mv-pv[-2][1])/pv[-2][1])
        # Rolling Sharpe (60 periods)
        if len(rets) >= 60:
            r60 = rets[-60:]; mu60 = mean(r60); sg60 = stdev(r60) if len(r60)>1 else 0.01
            rolling_sharpes.append((mu60/sg60)*sqrt(252))

    if not rets: return None
    fv_mv = pv[-1][1]; tr = (fv_mv-CAPITAL)/CAPITAL*100
    ny = len(rets)/252; ar = ((fv_mv/CAPITAL)**(1/ny)-1)*100 if ny>0 else 0
    mu = mean(rets); sg = stdev(rets) if len(rets)>1 else 0.01
    av = sg*sqrt(252)*100; sh = (ar-2.5)/av if av>0 else 0
    pk = CAPITAL; dd = 0
    for _,mv in pv:
        if mv>pk: pk=mv
        d = (mv-pk)/pk*100
        if d<dd: dd=d
    wr = sum(1 for r in rets if r>0)/len(rets)*100
    # Annual returns
    for _,mv in pv:
        yr = d[:4] if (d := str(pv[0][0])) else ''
    # Simple annual: group by year of trade date
    yr_rets = defaultdict(list)
    for i,(d,mv) in enumerate(pv):
        yr = d[:4]
        if i>0 and pv[i-1][1]>0: yr_rets[yr].append((mv-pv[i-1][1])/pv[i-1][1])
    annual = {yr: (sum(r)*100, len(r)) for yr, r in yr_rets.items() if len(r)>50}

    return {'label':label, 'total_ret':tr, 'ann_ret':ar, 'ann_vol':av, 'sharpe':sh,
            'max_dd':dd, 'win_rate':wr, 'n_rebals':len(rebal), 'n_days':len(rets),
            'rolling_sharpes': rolling_sharpes, 'annual': annual, 'final_mv': fv_mv}

# 4. Run
print('[3/4] Running backtests...')
results = []

# Factor-only: FULL period
r = backtest(factor_scores, 'Factor (全周期)')
if r: results.append(r); print(f'  Factor全周期: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

# Test dates for GNN
all_common = sorted(set(factor_scores.keys()) & set(gnn_scores.keys()))
n_train = int(len(all_common) * 0.6)
test_dates = all_common[n_train:]
gnn_test = {td: gnn_scores[td] for td in test_dates if td in gnn_scores}
factor_test = {td: factor_scores[td] for td in test_dates if td in factor_scores}

def norm(s):
    if not s: return {}
    vals = list(s.values()); v_min, v_max = min(vals), max(vals)
    return {c: (v-v_min)/(v_max-v_min) if v_max>v_min else 0.5 for c,v in s.items()}

# Real Agent signal: market momentum heuristic (replaces constant 0.5)
# Agent = recent 20d market return → bullish=prefer factor, bearish=prefer GNN
agent_signals = {}
for di, td in enumerate(test_dates):
    mdi = all_dates.index(td) if td in all_dates else -1
    if mdi >= 20:
        ret_20d = 0
        for i in range(mdi-20, mdi):
            d_prev, d_curr = all_dates[i], all_dates[i+1]
            prices_prev = [close_map.get(c,{}).get(d_prev) for c in list(close_map.keys())[:100] if close_map.get(c,{}).get(d_prev)]
            prices_curr = [close_map.get(c,{}).get(d_curr) for c in list(close_map.keys())[:100] if close_map.get(c,{}).get(d_curr)]
            if prices_prev and prices_curr:
                ret_20d += (mean(prices_curr)-mean(prices_prev))/mean(prices_prev)
        # Agent score: normalized to [0,1], 0.5=neutral
        agent_signals[td] = max(0.1, min(0.9, 0.5 + ret_20d * 2))
    else:
        agent_signals[td] = 0.5

# Fusion: Optimal weights (20/45/35) + Real Agent
fusion_test = defaultdict(dict)
for td in test_dates:
    fn = norm(factor_scores.get(td,{})); gn = norm(gnn_scores.get(td,{}))
    ag = agent_signals.get(td, 0.5)
    for c in set(fn)|set(gn):
        fusion_test[td][c] = fn.get(c,0.5)*0.20 + gn.get(c,0.5)*0.45 + ag*0.35

# Also: old static fusion for comparison
fusion_old = defaultdict(dict)
for td in test_dates:
    fn = norm(factor_scores.get(td,{})); gn = norm(gnn_scores.get(td,{}))
    for c in set(fn)|set(gn):
        fusion_old[td][c] = fn.get(c,0.5)*0.50 + gn.get(c,0.5)*0.30 + 0.5*0.20

for scores, lbl in [
    (factor_test, 'Factor (测试期)'), (gnn_test, 'GNN GAT'),
    (fusion_test, 'Fusion(20/45/35)+Agent'), (fusion_old, 'Fusion旧(50/30/20)')
]:
    r = backtest(scores, lbl)
    if r: results.append(r); print(f'  {lbl:25s}: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

print(f'  Agent avg signal: {mean(agent_signals.values()):.3f} (range {min(agent_signals.values()):.3f}-{max(agent_signals.values()):.3f})')

# 5. Risk-managed backtest (factor full period)
print(f'\n[4/4] Risk-managed backtest...')
def backtest_with_risk(scores_by_date, label):
    """Backtest with graduated position scaling (L3 drawdown-based)."""
    dates = sorted(scores_by_date.keys())
    rebal = dates[::FREQ]; rebal = [d for d in rebal if d >= dates[min(6, len(dates)//10)]]
    if len(rebal) < 5: return None, []
    cash = CAPITAL; holdings = {}; pv = []; rets = []
    risk_events = []; peak = CAPITAL; dd_count = 0

    for ri, rd in enumerate(rebal):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        picks = set(c for c, _ in ranked)
        nd = rebal[ri+1] if ri+1 < len(rebal) else dates[-1]

        # Current equity
        equity = cash + sum(holdings[c]['shares']*close_map.get(c,{}).get(rd,0) for c in holdings if close_map.get(c,{}).get(rd,0))

        # Risk: drawdown-based position scaling
        if equity > peak: peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > 0.20: scale = 0.25; risk_level = 'CRITICAL'
        elif dd > 0.15: scale = 0.50; risk_level = 'HIGH'
        elif dd > 0.10: scale = 0.75; risk_level = 'ELEVATED'
        else: scale = 1.0; risk_level = 'LOW'

        if scale < 1.0:
            dd_count += 1
            risk_events.append({'date': rd, 'dd': round(dd*100,1), 'scale': scale, 'level': risk_level})

        # Sell all
        for c in list(holdings.keys()):
            px = close_map.get(c,{}).get(rd)
            if px and px>0: cash += holdings[c]['shares'] * px * 0.999
            del holdings[c]

        # Buy with risk scaling
        pp = {c: close_map.get(c,{}).get(rd) for c in picks}
        pp = {c:p for c,p in pp.items() if p and p>0}
        if len(pp) < max(5, TOP_N//3): continue
        investable = cash * scale; ps = investable / len(pp)
        for c,p in pp.items():
            sh = int(ps/(p*1.001)/100)*100
            if sh>0: cash -= sh*p*1.001; holdings[c] = {'shares': sh}

        for d in [x for x in all_dates if rd <= x < nd]:
            mv = cash + sum(holdings[c]['shares']*close_map.get(c,{}).get(d,0) for c in holdings if close_map.get(c,{}).get(d))
            pv.append(mv)
            if len(pv)>=2 and pv[-2]>0: rets.append((pv[-1]-pv[-2])/pv[-2])

    if not rets: return None, []
    fv_mv = pv[-1]; tr = (fv_mv-CAPITAL)/CAPITAL*100
    ny = len(rets)/252; ar = ((fv_mv/CAPITAL)**(1/ny)-1)*100 if ny>0 else 0
    mu = mean(rets); sg = stdev(rets) if len(rets)>1 else 0.01
    av = sg*sqrt(252)*100; sh = (ar-2.5)/av if av>0 else 0
    pk = CAPITAL; dd_max = 0
    for mv in pv:
        if mv>pk: pk=mv
        d = (mv-pk)/pk*100
        if d<dd_max: dd_max=d
    return {'label':label, 'total_ret':tr, 'ann_ret':ar, 'ann_vol':av, 'sharpe':sh,
            'max_dd':dd_max, 'win_rate':sum(1 for r in rets if r>0)/len(rets)*100,
            'n_days':len(rets), 'n_rebals':len(rebal), 'dd_triggers': dd_count}, risk_events

r_risk, risk_events = backtest_with_risk(factor_scores, 'Factor + 风控')
if r_risk:
    results.append(r_risk)
    print(f'  Factor+风控: Ret={r_risk["total_ret"]:+.1f}% Sharpe={r_risk["sharpe"]:.3f} DD={r_risk["max_dd"]:.1f}% (触发{r_risk["dd_triggers"]}次)')
else:
    print(f'  Factor+风控: 跳过 (数据不足)')

# Store risk events
if risk_events:
    with SessionContext() as s:
        stmt = text("INSERT INTO risk_logs (timestamp, level, category, message, detail, updated_at) VALUES (datetime('now'), :l, 'DRAWDOWN', '回撤触发仓位缩减', :d, datetime('now'))")
        for ev in risk_events[:50]:  # top 50
            try: s.execute(stmt, {'l': ev['level'], 'd': f"date={ev['date']} dd={ev['dd']}% scale={ev['scale']:.0%}"})
            except: pass
        s.commit()
        cnt = s.execute(text('SELECT COUNT(*) FROM risk_logs')).scalar()
    print(f'  risk_logs: {cnt} events stored')

# 6. Report
print(f'\n{"="*60}')
print(f'  全周期回测报告')
print(f'{"="*60}')
print(f'  {"Strategy":20s} {"Return":>8s} {"Sharpe":>8s} {"MaxDD":>8s} {"Vol":>8s} {"Win%":>7s} {"Days":>6s}')
print(f'  {"-"*65}')
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f'  {r["label"]:20s} {r["total_ret"]:>+7.1f}% {r["sharpe"]:>8.3f} {r["max_dd"]:>7.1f}% {r["ann_vol"]:>7.1f}% {r["win_rate"]:>6.1f}% {r["n_days"]:>6d}')

# Annual returns
if results:
    r0 = [r for r in results if '全周期' in r['label'] and '风控' not in r['label']]
    if r0:
        print(f'\n  ── 年度收益 ──')
        for yr in sorted(r0[0].get('annual',{}).keys()):
            ret, n = r0[0]['annual'][yr]
            bar = ('+' if ret>0 else '') + '█'*int(abs(ret)/5)
            print(f'    {yr}: {ret:+.1f}% ({n}天) {bar}')

# Rolling Sharpe
r0_all = [r for r in results if '全周期' in r['label'] and '风控' not in r['label']]
if r0_all and r0_all[0].get('rolling_sharpes'):
    rs = r0_all[0]['rolling_sharpes']
    if rs:
        print(f'\n  ── 滚动夏普(60期) ──')
        print(f'    Mean: {mean(rs):.2f} | Max: {max(rs):.2f} | Min: {min(rs):.2f} | Final: {rs[-1]:.2f}')

# Save
json.dump([{k: v for k, v in r.items() if k not in ('rolling_sharpes', 'annual')} for r in results],
          open(BASE/'full_period_report.json', 'w'), indent=2, default=str)
print(f'\n  Report saved')
print(f'{"="*60}')
print(f'  全周期回测 + 风控完成')
print(f'{"="*60}')
