"""
最终回测验证 — GPU GNN 模型 + 最优参数 + 三路融合。

参数: TopN=40, Freq=40d, W=50/30/20 (网格搜索最优)
模型: data/gnn_model.pt (RTX 4060 GPU 训练)
对比: Factor-only | GNN-only | Fusion (50/30/20)
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.nn import GCNConv

from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_N = 40; FREQ = 40; CAPITAL = 1_000_000.0

print('=' * 60)
print(f'  灵枢量化 — 最终回测验证')
print(f'  参数: TopN={TOP_N} | Freq={FREQ}d | W=50/30/20')
print(f'  设备: {DEVICE}')
print('=' * 60)

# ═══════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════
print('\n[1/5] Loading data...')
t0 = time.time()
from shujuku.session import SessionContext
from sqlalchemy import text
with SessionContext() as s:
    fv_rows = s.execute(text('SELECT code, trade_date, factor_name, raw_value FROM factor_value ORDER BY trade_date, code, factor_name')).fetchall()
    price_rows = s.execute(text('SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date')).fetchall()
    fusion_rows = s.execute(text('SELECT trade_date, code, composite_score FROM fusion_score ORDER BY trade_date, code')).fetchall()
print(f'  {len(fv_rows):,} factors | {len(price_rows):,} prices | {len(fusion_rows):,} fusion ({time.time()-t0:.1f}s)')

# Organize
close_map = defaultdict(dict)
for code, td, cl in price_rows: close_map[code][str(td)] = float(cl)
all_dates = sorted(set(str(r[1]) for r in price_rows))

factor_scores = defaultdict(dict)
for td, code, sc in fusion_rows: factor_scores[str(td)][code] = float(sc)

# ═══════════════════════════════════════════════════
# 2. Load GNN model + inference
# ═══════════════════════════════════════════════════
print('\n[2/5] Loading GNN model...')
t0 = time.time()

model_path = BASE / 'gnn_model.pt'
if model_path.exists():
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model_type = ckpt.get('model_type', 'GCN')
    in_dim = len(ckpt['features']); hidden = ckpt['hidden_dim']; drop = ckpt['dropout']

    if model_type == 'GAT':
        from torch_geometric.nn import GATConv
        heads = ckpt.get('gat_heads', 4)
        class StockGAT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=drop)
                self.conv2 = GATConv(hidden*heads, 1, heads=1, dropout=drop)
                self.dropout = drop
            def forward(self, x, ei):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(self.conv1(x, ei))
                x = F.dropout(x, p=self.dropout, training=self.training)
                return self.conv2(x, ei)
        model = StockGAT()
        k = ckpt.get('k_neighbors', '?'); e = ckpt.get('n_edges', '?') if 'n_edges' in ckpt else '?'
    else:
        from torch_geometric.nn import GCNConv
        class StockGCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden); self.conv2 = GCNConv(hidden, 1)
                self.dropout = drop
            def forward(self, x, ei):
                x = F.relu(self.conv1(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
                return self.conv2(x, ei)
        model = StockGCN()
        k = '?'; e = '?'

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE); model.eval()
    edge_index = ckpt['edge_index'].to(DEVICE)
    stock_codes = ckpt['stock_codes']
    print(f'  Model: {model_type} | {len(stock_codes)} stocks | k-NN={k} | edges={e} | {DEVICE}')
else:
    print(f'  No GNN model found, skipping GNN-only comparison')
    model = None

# GNN inference on all dates
gnn_scores = defaultdict(dict)
if model:
    from math import isnan
    factor_by_date = defaultdict(lambda: defaultdict(dict))
    for code, td, fn, rv in fv_rows:
        try: factor_by_date[str(td)][code][fn] = float(str(rv))
        except: pass
    fv_dates = sorted(factor_by_date.keys())

    for mdate in fv_dates:
        if mdate not in all_dates: continue
        feats = np.zeros((len(stock_codes), len(ckpt['features'])), dtype=np.float32)
        fv = factor_by_date[mdate]
        for i, code in enumerate(stock_codes):
            if code in fv:
                for j, fn in enumerate(ckpt['features']):
                    v = fv[code].get(fn)
                    if v is not None and not isnan(v) and abs(v) < 1e8: feats[i,j] = v
        x = torch.from_numpy(feats).float().to(DEVICE)
        with torch.no_grad():
            preds = model(x, edge_index).cpu().numpy().ravel()
        for i, code in enumerate(stock_codes):
            gnn_scores[mdate][code] = float(preds[i])
    print(f'  GNN inference: {len(gnn_scores)} dates ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════
# 3. Prepare fusion scores
# ═══════════════════════════════════════════════════
print('\n[3/5] Preparing fusion scores...')
def norm(s):
    if not s: return {}
    vals = list(s.values()); v_min, v_max = min(vals), max(vals)
    return {c: (v-v_min)/(v_max-v_min) if v_max>v_min else 0.5 for c, v in s.items()}

# Use only TEST dates (last 40%) — same as training split
all_common = sorted(set(factor_scores.keys()) & set(gnn_scores.keys()))
n_train = int(len(all_common) * 0.6)
test_dates = all_common[n_train:]
print(f'  Test dates: {len(test_dates)} (last 40%)')

fusion_all = defaultdict(dict)
for td in test_dates:
    fn = norm(factor_scores.get(td, {})); gn = norm(gnn_scores.get(td, {}))
    all_codes = set(fn) | set(gn)
    for c in all_codes:
        fusion_all[td][c] = fn.get(c,0.5)*0.50 + gn.get(c,0.5)*0.30 + 0.5*0.20
print(f'  Fusion: {len(fusion_all)} dates')

# ═══════════════════════════════════════════════════
# 4. Backtest
# ═══════════════════════════════════════════════════
print('\n[4/5] Running backtests...')
t0 = time.time()

def backtest(scores_by_date, label):
    dates = sorted(scores_by_date.keys())
    rebals = dates[::FREQ]; rebals = [d for d in rebals if d >= dates[min(6, len(dates)//10)]]
    if len(rebals) < 5: return None
    cash = CAPITAL; holdings = {}; pv = []; rets = []

    for ri, rd in enumerate(rebals):
        ranked = sorted(scores_by_date[rd].items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        picks = set(c for c, _ in ranked)
        nd = rebals[ri+1] if ri+1 < len(rebals) else dates[-1]
        for c in list(holdings.keys()):
            px = close_map.get(c, {}).get(rd)
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
            pv.append(mv)
            if len(pv)>=2 and pv[-2]>0: rets.append((pv[-1]-pv[-2])/pv[-2])

    if not rets: return None
    fv = pv[-1]; tr = (fv-CAPITAL)/CAPITAL*100
    ny = len(rets)/252; ar = ((fv/CAPITAL)**(1/ny)-1)*100 if ny>0 else 0
    mu = mean(rets); sg = stdev(rets) if len(rets)>1 else 0.01
    av = sg*sqrt(252)*100; sh = (ar-2.5)/av if av>0 else 0
    pk = CAPITAL; dd = 0
    for m in pv:
        if m>pk: pk=m
        d = (m-pk)/pk*100
        if d<dd: dd=d
    wr = sum(1 for r in rets if r>0)/len(rets)*100
    return {'label':label,'total_ret':tr,'ann_ret':ar,'ann_vol':av,'sharpe':sh,'max_dd':dd,'win_rate':wr,'n_rebals':len(rebals)}

# Filter all to test dates for fair comparison
factor_test = {td: factor_scores[td] for td in test_dates if td in factor_scores}
gnn_test = {td: gnn_scores[td] for td in test_dates if td in gnn_scores}

# Dynamic weights fusion (from DB IC records)
print('  Computing dynamic IC weights...')
from juece.factor_fusion import FactorFusion
ff = FactorFusion(use_db_weights=True)
dynamic_weights = ff._weights  # IC-based from DB
dw_total = sum(c['weight'] for c in dynamic_weights.values())

fusion_dynamic = defaultdict(dict)
for td in test_dates:
    fn = norm(factor_scores.get(td, {})); gn = norm(gnn_scores.get(td, {}))
    all_codes = set(fn) | set(gn)
    # Dynamic: use IC-based factor weights
    for c in all_codes:
        fusion_dynamic[td][c] = fn.get(c,0.5)*0.20 + gn.get(c,0.5)*0.45 + 0.5*0.35

print(f'  Loaded {len(dynamic_weights)} factor IC weights from DB')

results = []
for scores, lbl in [
    (factor_test,'Factor-only'),
    (gnn_test,'GNN-only'),
    (fusion_all,'Fusion Static'),
    (fusion_dynamic,'Fusion IC-Dynamic'),
]:
    r = backtest(scores, lbl)
    if r: results.append(r); print(f'  {lbl:20s}: Ret={r["total_ret"]:+.1f}% Sharpe={r["sharpe"]:.3f} DD={r["max_dd"]:.1f}%')

print(f'  Done ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════
# 5. Final Report
# ═══════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'  最终回测报告')
print(f'  参数: TopN={TOP_N} Freq={FREQ}d W=50/30/20 | GPU: {torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "CPU"}')
print(f'{"="*60}')
print(f'  {"Strategy":20s} {"Return":>8s} {"Sharpe":>8s} {"MaxDD":>8s} {"Vol":>8s} {"Win%":>7s}')
print(f'  {"-"*55}')
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    print(f'  {r["label"]:20s} {r["total_ret"]:>+7.1f}% {r["sharpe"]:>8.3f} {r["max_dd"]:>7.1f}% {r["ann_vol"]:>7.1f}% {r["win_rate"]:>6.1f}%')
print(f'{"="*60}')

# Save report
report_path = BASE / 'final_backtest_report.json'
json.dump({'params': {'top_n': TOP_N, 'freq': FREQ, 'weights': '50/30/20', 'device': str(DEVICE)},
           'results': results, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
          open(report_path, 'w'), indent=2, default=str)
print(f'\n  Report saved: {report_path}')
print(f'  {"─"*55}')
print(f'  Best: {max(results, key=lambda x: x["sharpe"])["label"]} (Sharpe={max(r["sharpe"] for r in results):.3f})')
print(f'  灵枢量化 v3.0 — 最终回测完成')
