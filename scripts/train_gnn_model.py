"""GNN模型训练+持久化 — GAT + 稀疏k-NN图 + Ranking Loss + GPU。
修复: 1)图稀疏化(k-NN top10) 2)GCN→GAT多头注意力
输出: data/gnn_model.pt, data/gnn_config.json, data/gnn_predictions.json"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict
from math import isnan
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.nn import GATConv
from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FORWARD_DAYS=20; HIDDEN_DIM=64; DROPOUT=0.3; EPOCHS=100; LR=0.005
TOP_FACTORS=['roe','roa','net_margin','gross_margin','pb','ps','pe',
    'momentum_3m','momentum_6m','momentum_12m1m','momentum_1m',
    'oc_20','cntp_20','corr_20','roc_5',
    'std_20','historical_vol','min_20','max_20','turn_20']
K_NEIGHBORS = 10  # 稀疏化: 每个节点只连最近10个邻居
GAT_HEADS = 4     # 多头注意力

print(f'GNN Training | Device: {DEVICE} | Model: GAT({GAT_HEADS}heads) | Graph: k-NN(k={K_NEIGHBORS})')
if DEVICE.type=='cuda': print(f'GPU: {torch.cuda.get_device_name(0)}')
t_total = time.time()

# 1. Load data
print('[1/6] Loading...'); t0=time.time()
from shujuku.session import SessionContext
from sqlalchemy import text
with SessionContext() as s:
    fv_rows = s.execute(text('SELECT code,trade_date,factor_name,raw_value FROM factor_value ORDER BY trade_date,code,factor_name')).fetchall()
    pr_rows = s.execute(text('SELECT code,trade_date,close FROM daily_bar ORDER BY code,trade_date')).fetchall()
    ind_rows = s.execute(text('SELECT code,sw_level1 FROM industry_classification')).fetchall()
print(f'  {len(fv_rows):,} factors | {len(pr_rows):,} prices | {len(ind_rows):,} industries ({time.time()-t0:.1f}s)')

# Organize
industry_map = {r[0]: r[1] for r in ind_rows}
stock_codes = sorted(set(r[0] for r in fv_rows))
stock_to_idx = {c: i for i, c in enumerate(stock_codes)}

close_map = defaultdict(dict)
for code, td, cl in pr_rows: close_map[code][str(td)] = float(cl)
all_dates = sorted(set(str(r[1]) for r in pr_rows))

factor_by_date = defaultdict(lambda: defaultdict(dict))
for code, td, fn, rv in fv_rows:
    try: factor_by_date[str(td)][code][fn] = float(str(rv))
    except: pass
fv_dates = sorted(factor_by_date.keys())

# 2. Build SPARSE k-NN correlation graph (fix 1: 稀疏化)
print(f'[2/6] Building sparse k-NN graph (k={K_NEIGHBORS})...'); t0=time.time()

# Compute average correlation between stocks using factor values
# Use all dates to build a robust correlation matrix
print(f'  Computing correlations...')
n_stocks = len(stock_codes)
# Collect factor vectors per stock (mean across time, then correlate)
stock_feat_matrix = np.zeros((n_stocks, len(TOP_FACTORS)), dtype=np.float32)
count_matrix = np.zeros(n_stocks, dtype=np.int32)

for mdate in fv_dates[:200]:  # Use first 200 dates for graph construction
    fv = factor_by_date[mdate]
    for i, code in enumerate(stock_codes):
        if code in fv:
            for j, fn in enumerate(TOP_FACTORS):
                v = fv[code].get(fn)
                if v is not None and not isnan(v) and abs(v) < 1e8:
                    stock_feat_matrix[i, j] += v
                    count_matrix[i] += 1

# Average
for i in range(n_stocks):
    if count_matrix[i] > 0:
        stock_feat_matrix[i] /= count_matrix[i]

# Compute pairwise correlation and build k-NN edges
from scipy.spatial.distance import cdist
# Normalize
mu = stock_feat_matrix.mean(axis=0); sg = stock_feat_matrix.std(axis=0) + 1e-12
feat_norm = (stock_feat_matrix - mu) / sg

# Compute pairwise distances and find k nearest neighbors
edges = []
for i in range(n_stocks):
    # Distance to all others
    diff = feat_norm - feat_norm[i]
    dist = np.sqrt((diff ** 2).sum(axis=1))
    dist[i] = 1e9  # exclude self
    # Top k nearest
    neighbors = np.argpartition(dist, K_NEIGHBORS)[:K_NEIGHBORS]
    for j in neighbors:
        if j != i:
            edges.append((i, int(j)))
            edges.append((int(j), i))  # symmetric

# Add industry edges (max 5 per stock)
ind_groups = defaultdict(list)
for i, code in enumerate(stock_codes):
    ind = industry_map.get(code, '其他')
    ind_groups[ind].append(i)

for ind, members in ind_groups.items():
    for i in members:
        others = [m for m in members if m != i]
        if len(others) > 5:
            others = list(np.random.choice(others, 5, replace=False))
        for j in others:
            edges.append((i, j))

# Deduplicate
edges = list(set(edges))
edge_index = torch.tensor([[s, d] for s, d in edges], dtype=torch.long).t().contiguous()
print(f'  {n_stocks} nodes | {len(edges):,} edges ({len(edges)//n_stocks} per stock) | ({time.time()-t0:.1f}s)')

# 3. Prepare features
print('[3/6] Features...'); t0=time.time()
features_list, labels_list, dates_list = [], [], []
for mdate in fv_dates:
    if mdate not in all_dates: continue
    mdi = all_dates.index(mdate); fi = min(mdi+FORWARD_DAYS, len(all_dates)-1); fd = all_dates[fi]
    feats = np.zeros((n_stocks, len(TOP_FACTORS)), dtype=np.float32)
    labels = np.full(n_stocks, np.nan, dtype=np.float32)
    fv = factor_by_date[mdate]
    for i, code in enumerate(stock_codes):
        if code in fv:
            for j, fn in enumerate(TOP_FACTORS):
                v = fv[code].get(fn)
                if v is not None and not isnan(v) and abs(v) < 1e8: feats[i,j] = v
    for i, code in enumerate(stock_codes):
        p0 = close_map.get(code,{}).get(mdate); p1 = close_map.get(code,{}).get(fd)
        if p0 and p1 and p0 > 0: labels[i] = (p1-p0)/p0
    if (~np.isnan(labels)).sum() < 50: continue
    features_list.append(feats); labels_list.append(labels); dates_list.append(mdate)
print(f'  {len(dates_list)} snapshots ({time.time()-t0:.1f}s)')

# 4. Train GAT with Ranking Loss (fix 2: GCN→GAT)
print(f'[4/6] Training GAT({GAT_HEADS} heads, k={K_NEIGHBORS})...'); t0=time.time()

class StockGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(len(TOP_FACTORS), HIDDEN_DIM, heads=GAT_HEADS, dropout=DROPOUT)
        self.conv2 = GATConv(HIDDEN_DIM * GAT_HEADS, 1, heads=1, dropout=DROPOUT)
        self.dropout = DROPOUT
    def forward(self, x, ei):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, ei)

def ranking_loss(preds, labels, margin=0.02, max_pairs=500):
    n = len(preds)
    if n < 2: return torch.tensor(0.0, device=preds.device)
    idx = torch.randperm(n, device=preds.device)[:max_pairs*2]
    if len(idx) < 2: return torch.tensor(0.0, device=preds.device)
    h = len(idx)//2; i_idx = idx[:h]; j_idx = idx[h:2*h]
    ld = labels[i_idx] - labels[j_idx]; pp = ld > 0
    if pp.sum() == 0: return torch.tensor(0.0, device=preds.device)
    return torch.clamp(margin - (preds[i_idx][pp] - preds[j_idx][pp]), min=0).mean()

n_train = int(len(dates_list) * 0.6)
model = StockGAT().to(DEVICE); edge_index = edge_index.to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

for epoch in range(1, EPOCHS+1):
    model.train(); tl, nb = 0, 0
    for t in range(n_train):
        x = torch.from_numpy(features_list[t]).float().to(DEVICE)
        y = torch.from_numpy(labels_list[t]).float().view(-1).to(DEVICE)
        vm = ~torch.isnan(y)
        if vm.sum() < 50: continue
        preds = model(x, edge_index).view(-1)
        loss = ranking_loss(preds[vm], y[vm])
        if loss.item() == 0: continue
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item(); nb += 1
    if epoch % 20 == 0 and nb > 0: print(f'  Epoch {epoch:3d}: rank_loss={tl/nb:.6f}')

# Save
model_path = BASE / 'gnn_model.pt'
cpu_w = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
torch.save({'model_state_dict': cpu_w, 'features': TOP_FACTORS, 'hidden_dim': HIDDEN_DIM,
    'dropout': DROPOUT, 'stock_codes': stock_codes, 'edge_index': edge_index.cpu(),
    'model_type': 'GAT', 'gat_heads': GAT_HEADS, 'k_neighbors': K_NEIGHBORS}, model_path)
json.dump({'features': TOP_FACTORS, 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
    'n_stocks': n_stocks, 'n_train': n_train, 'n_total': len(dates_list),
    'loss': 'pairwise_ranking', 'model': 'GAT', 'heads': GAT_HEADS,
    'k_neighbors': K_NEIGHBORS, 'n_edges': len(edges)},
    open(BASE/'gnn_config.json', 'w'), indent=2)
print(f'  Saved: {model_path} ({model_path.stat().st_size/1024:.0f}KB)')

# 5. Inference
print('[5/6] Inference...'); t0=time.time()
model.eval(); all_preds = {}
for t in range(len(dates_list)):
    x = torch.from_numpy(features_list[t]).float().to(DEVICE)
    with torch.no_grad(): preds = model(x, edge_index).cpu().numpy().ravel()
    for i, code in enumerate(stock_codes):
        if not np.isnan(labels_list[t][i]):
            all_preds[f'{dates_list[t]}|{code}'] = float(preds[i])
    if (t+1) % 200 == 0: print(f'  {t+1}/{len(dates_list)}')
pred_path = BASE / 'gnn_predictions.json'
json.dump(all_preds, open(pred_path, 'w'))
print(f'  Saved: {pred_path} ({pred_path.stat().st_size/1024/1024:.0f}MB, {len(all_preds):,} entries)')

# 6. Evaluation
print(f'[6/6] Evaluation...'); t0=time.time()
# Top-K precision on test set
from statistics import mean
test_start = n_train
precisions = {5: [], 10: [], 30: []}
ndcgs = []

for t in range(test_start, len(dates_list)):
    preds = np.array([all_preds.get(f'{dates_list[t]}|{c}', 0) for c in stock_codes])
    labels = labels_list[t]
    valid = ~np.isnan(labels)
    if valid.sum() < 50: continue

    # Top-K precision: % of top-K predicted that are in top-K actual
    true_top30 = set(stock_codes[i] for i in np.argsort(labels)[-30:])
    for k in [5, 10, 30]:
        pred_top = set(stock_codes[i] for i in np.argsort(preds)[-k:])
        precisions[k].append(len(pred_top & true_top30) / k)

if any(precisions[k] for k in precisions):
    print(f'  Top-K Precision (test set):')
    for k in [5, 10, 30]:
        if precisions[k]:
            print(f'    Top-{k}: {mean(precisions[k]):.3f}')

print(f'\nDone: {time.time()-t_total:.0f}s total')
