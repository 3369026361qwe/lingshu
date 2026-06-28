"""搜索最优融合权重 (Factor/GNN/Agent) — 使用真实 GAT 模型"""
import sys
import time
from collections import defaultdict
from math import isnan, sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_N = 40; FREQ = 40; CAPITAL = 1_000_000.0

print('=' * 60)
print('  Fusion Weight Grid Search (Real GAT)')
print('=' * 60)

# 1. Load
print('[1/4] Loading...'); t0 = time.time()
from sqlalchemy import text

from shujuku.session import SessionContext

with SessionContext() as s:
    pr = s.execute(text('SELECT code,trade_date,close FROM daily_bar ORDER BY code,trade_date')).fetchall()
    fs = s.execute(text('SELECT trade_date,code,composite_score FROM fusion_score ORDER BY trade_date,code')).fetchall()
print(f'  {len(pr):,} prices | {len(fs):,} fusion ({time.time()-t0:.1f}s)')

close_map = defaultdict(dict)
for c,td,cl in pr: close_map[c][str(td)] = float(cl)
all_dates = sorted(set(str(r[1]) for r in pr))
factor_scores = defaultdict(dict)
for td,c,sc in fs: factor_scores[str(td)][c] = float(sc)

# 2. Load GNN
print('[2/4] Loading GNN...'); t0 = time.time()
ckpt = torch.load(BASE/'gnn_model.pt', map_location='cpu', weights_only=True)
mt = ckpt.get('model_type','GCN')
in_dim = len(ckpt['features']); hd = ckpt['hidden_dim']; dp = ckpt['dropout']
stock_codes = ckpt['stock_codes']; edge_index = ckpt['edge_index'].to(DEVICE)

if mt == 'GAT':
    from torch_geometric.nn import GATConv
    heads = ckpt.get('gat_heads',4)
    class GNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(in_dim, hd, heads=heads, dropout=dp)
            self.conv2 = GATConv(hd*heads, 1, heads=1, dropout=dp)
        def forward(self, x, ei):
            x = F.dropout(x, p=dp, training=False)
            x = F.elu(self.conv1(x,ei))
            x = F.dropout(x, p=dp, training=False)
            return self.conv2(x,ei)
else:
    from torch_geometric.nn import GCNConv
    class GNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hd); self.conv2 = GCNConv(hd, 1)
        def forward(self, x, ei):
            x = F.relu(self.conv1(x,ei))
            x = F.dropout(x, p=dp, training=False)
            return self.conv2(x,ei)

model = GNNModel().to(DEVICE); model.load_state_dict(ckpt['model_state_dict']); model.eval()

with SessionContext() as s:
    fv_rows = s.execute(text('SELECT code,trade_date,factor_name,raw_value FROM factor_value ORDER BY trade_date,code,factor_name')).fetchall()
factor_by_date = defaultdict(lambda: defaultdict(dict))
for c,td,fn,rv in fv_rows:
    try: factor_by_date[str(td)][c][fn] = float(str(rv))
    except: pass
fv_dates = sorted(factor_by_date.keys())

gnn_scores = defaultdict(dict)
for mdate in fv_dates:
    if mdate not in all_dates: continue
    f = np.zeros((len(stock_codes), len(ckpt['features'])), dtype=np.float32)
    fv_d = factor_by_date[mdate]
    for i,c in enumerate(stock_codes):
        if c in fv_d:
            for j,fn in enumerate(ckpt['features']):
                v = fv_d[c].get(fn)
                if v is not None and not isnan(v) and abs(v)<1e8: f[i,j] = v
    x = torch.from_numpy(f).float().to(DEVICE)
    with torch.no_grad(): p = model(x, edge_index).cpu().numpy().ravel()
    for i,c in enumerate(stock_codes): gnn_scores[mdate][c] = float(p[i])
print(f'  {mt} | {len(gnn_scores)} dates ({time.time()-t0:.1f}s)')

# 3. Prepare test dates
all_common = sorted(set(factor_scores.keys()) & set(gnn_scores.keys()))
n_train = int(len(all_common) * 0.6)
test_dates = all_common[n_train:]
print(f'  Test dates: {len(test_dates)}')

def norm(s):
    if not s: return {}
    vals = list(s.values()); v_min, v_max = min(vals), max(vals)
    return {c: (v-v_min)/(v_max-v_min) if v_max>v_min else 0.5 for c,v in s.items()}

fn_norm = {td: norm(factor_scores.get(td,{})) for td in test_dates}
gn_norm = {td: norm(gnn_scores.get(td,{})) for td in test_dates}

# 4. Backtest
def run_bt(scores_by_date):
    dates = sorted(scores_by_date.keys())
    rebal = dates[::FREQ]; rebal = [d for d in rebal if d >= dates[min(6, len(dates)//10)]]
    if len(rebal) < 5: return None
    cash = CAPITAL; holdings = {}; pv = []; rets = []
    for ri, rd in enumerate(rebal):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        picks = set(c for c,_ in ranked)
        nd = rebal[ri+1] if ri+1 < len(rebal) else dates[-1]
        for c in list(holdings.keys()):
            px = close_map.get(c,{}).get(rd)
            if px and px>0: cash += holdings[c]['shares']*px*0.999
            del holdings[c]
        pp = {c: close_map.get(c,{}).get(rd) for c in picks}
        pp = {c:p for c,p in pp.items() if p and p>0}
        if len(pp) < max(5, TOP_N//3): continue
        ps = cash/len(pp)
        for c,p in pp.items():
            sh = int(ps/(p*1.001)/100)*100
            if sh>0: cash -= sh*p*1.001; holdings[c] = {'shares': sh}
        for d in [x for x in all_dates if rd <= x < nd]:
            mv = cash + sum(holdings[c]['shares']*close_map.get(c,{}).get(d,0) for c in holdings if close_map.get(c,{}).get(d))
            pv.append(mv)
            if len(pv)>=2 and pv[-2]>0: rets.append((pv[-1]-pv[-2])/pv[-2])
    if not rets: return None
    fv_mv = pv[-1]; tr = (fv_mv-CAPITAL)/CAPITAL*100
    ny = len(rets)/252; ar = ((fv_mv/CAPITAL)**(1/ny)-1)*100 if ny>0 else 0
    mean(rets); sg = stdev(rets) if len(rets)>1 else 0.01
    av = sg*sqrt(252)*100; sh = (ar-2.5)/av if av>0 else 0
    pk = CAPITAL; dd = 0
    for mv in pv:
        if mv>pk: pk=mv
        d = (mv-pk)/pk*100
        if d<dd: dd=d
    return {'ret': tr, 'sharpe': sh, 'dd': dd, 'vol': av}

# 5. Grid search
print('[3/4] Grid search...')
print(f'  {"F%":>5s} {"G%":>5s} {"A%":>5s} {"Sharpe":>8s} {"Return":>8s} {"MaxDD":>8s}')

combos = []
for fw in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    for gw in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        aw = 1.0 - fw - gw
        if aw < 0.05 or aw > 0.4: continue
        fusion = defaultdict(dict)
        for td in test_dates:
            fn = fn_norm.get(td,{}); gn = gn_norm.get(td,{})
            for c in set(fn)|set(gn):
                fusion[td][c] = fn.get(c,0.5)*fw + gn.get(c,0.5)*gw + 0.5*aw
        r = run_bt(fusion)
        if r: combos.append({'fw':fw, 'gw':gw, 'aw':aw, **r})

combos.sort(key=lambda x: x['sharpe'], reverse=True)
for c in combos[:15]:
    print(f'  {c["fw"]:>.0%} {c["gw"]:>.0%} {c["aw"]:>.0%} {c["sharpe"]:>8.3f} {c["ret"]:>+7.1f}% {c["dd"]:>7.1f}%')

# Baseline
r_f = run_bt({td: fn_norm[td] for td in test_dates if td in fn_norm})
r_g = run_bt({td: gn_norm[td] for td in test_dates if td in gn_norm})
cur = [c for c in combos if c['fw']==0.5 and c['gw']==0.3]

print('\n[4/4] Summary')
print(f'  Factor-only:       Sharpe={r_f["sharpe"]:.3f} Ret={r_f["ret"]:+.1f}% DD={r_f["dd"]:.1f}%')
print(f'  GNN-only:          Sharpe={r_g["sharpe"]:.3f} Ret={r_g["ret"]:+.1f}% DD={r_g["dd"]:.1f}%')
print(f'  Current(50/30/20): Sharpe={cur[0]["sharpe"]:.3f} Ret={cur[0]["ret"]:+.1f}% DD={cur[0]["dd"]:.1f}%' if cur else '  Current: N/A')
best = combos[0]
print(f'  Best Fusion:       Sharpe={best["sharpe"]:.3f} Ret={best["ret"]:+.1f}% DD={best["dd"]:.1f}%')
print(f'  Best weights:      F={best["fw"]:.0%} G={best["gw"]:.0%} A={best["aw"]:.0%}')
print(f'{"="*60}')
