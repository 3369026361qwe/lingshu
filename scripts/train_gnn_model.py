"""GNN模型训练 — GAT + 稀疏k-NN图 + Ranking Loss + GPU。

使用 tushenjing 模块的 graph_utils.build_knn_graph / ranking_loss 和
graph_trainer.GraphTrainer.fit_batches()，消除内联模型定义和训练循环。

输出: data/gnn_model.pt, data/gnn_config.json, data/gnn_predictions.json
"""
import argparse
import json
import time
from collections import defaultdict
from math import isnan
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.nn import GATConv

from tushenjing.graph_trainer import GraphTrainer
from tushenjing.graph_utils import GraphUtils

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

BASE = Path('E:/28721/lingshu/data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FORWARD_DAYS = 20
HIDDEN_DIM = 64
DROPOUT = 0.3
EPOCHS = 100
LR = 0.005
K_NEIGHBORS = 10
GAT_HEADS = 4

TOP_FACTORS = [
    'roe', 'roa', 'net_margin', 'gross_margin', 'pb', 'ps', 'pe',
    'momentum_3m', 'momentum_6m', 'momentum_12m1m', 'momentum_1m',
    'oc_20', 'cntp_20', 'corr_20', 'roc_5',
    'std_20', 'historical_vol', 'min_20', 'max_20', 'turn_20',
]


class StockGAT(torch.nn.Module):
    """GAT 模型。定义在此以支持 torch.save 完整序列化。"""
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(len(TOP_FACTORS), HIDDEN_DIM, heads=GAT_HEADS, dropout=DROPOUT)
        self.conv2 = GATConv(HIDDEN_DIM * GAT_HEADS, 1, heads=1, dropout=DROPOUT)

    def forward(self, x, ei):
        x = torch.nn.functional.dropout(x, p=DROPOUT, training=self.training)
        x = torch.nn.functional.elu(self.conv1(x, ei))
        x = torch.nn.functional.dropout(x, p=DROPOUT, training=self.training)
        return self.conv2(x, ei)


def main():
    parser = argparse.ArgumentParser(description='GNN 模型训练')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--k-neighbors', type=int, default=K_NEIGHBORS)
    parser.add_argument('--hidden-dim', type=int, default=HIDDEN_DIM)
    args = parser.parse_args()

    print(f'GNN Training | {DEVICE} | GAT({GAT_HEADS}heads) | k-NN(k={args.k_neighbors})')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    t_total = time.time()

    # 1 ── 加载数据 ──
    print('[1/6] Loading...')
    t0 = time.time()
    from sqlalchemy import text

    from shujuku.session import SessionContext
    with SessionContext() as s:
        fv_rows = s.execute(text(
            'SELECT code, trade_date, factor_name, raw_value '
            'FROM factor_value ORDER BY trade_date, code, factor_name'
        )).fetchall()
        pr_rows = s.execute(text(
            'SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date'
        )).fetchall()
        ind_rows = s.execute(text(
            'SELECT code, sw_level1 FROM industry_classification'
        )).fetchall()
    print(f'  {len(fv_rows):,} factors | {len(pr_rows):,} prices | {len(ind_rows):,} industries ({time.time() - t0:.1f}s)')

    industry_map = {r[0]: r[1] for r in ind_rows}
    stock_codes = sorted(set(r[0] for r in fv_rows))

    close_map = defaultdict(dict)
    for code, td, cl in pr_rows:
        close_map[code][str(td)] = float(cl)
    all_dates = sorted(set(str(r[1]) for r in pr_rows))
    date_index = {d: i for i, d in enumerate(all_dates)}

    factor_by_date = defaultdict(lambda: defaultdict(dict))
    for code, td, fn, rv in fv_rows:
        try:
            factor_by_date[str(td)][code][fn] = float(str(rv))
        except Exception:
            pass
    fv_dates = sorted(factor_by_date.keys())

    # 2 ── 构建 k-NN 图 (使用模块) ──
    print(f'[2/6] Building sparse k-NN graph (k={args.k_neighbors})...')
    t0 = time.time()
    n_stocks = len(stock_codes)
    edges = GraphUtils.build_knn_graph(
        stock_codes, factor_by_date, TOP_FACTORS, industry_map,
        k_neighbors=args.k_neighbors, n_dates=200,
    )
    edge_index = torch.from_numpy(np.array(edges, dtype=np.int64)).t().contiguous()
    print(f'  {n_stocks} nodes | {len(edges):,} edges ({len(edges) // n_stocks}/stock) | ({time.time() - t0:.1f}s)')

    # 3 ── 构建特征/标签 ──
    print('[3/6] Features...')
    t0 = time.time()
    features_list, labels_list, dates_list = [], [], []
    for mdate in fv_dates:
        if mdate not in all_dates:
            continue
        mdi = date_index[mdate]
        fi = min(mdi + FORWARD_DAYS, len(all_dates) - 1)
        fd = all_dates[fi]
        feats = np.zeros((n_stocks, len(TOP_FACTORS)), dtype=np.float32)
        labels = np.full(n_stocks, np.nan, dtype=np.float32)
        fv = factor_by_date[mdate]
        for i, code in enumerate(stock_codes):
            if code in fv:
                for j, fn in enumerate(TOP_FACTORS):
                    v = fv[code].get(fn)
                    if v is not None and not isnan(v) and abs(v) < 1e8:
                        feats[i, j] = v
        for i, code in enumerate(stock_codes):
            p0 = close_map.get(code, {}).get(mdate)
            p1 = close_map.get(code, {}).get(fd)
            if p0 and p1 and p0 > 0:
                labels[i] = (p1 - p0) / p0
        if (~np.isnan(labels)).sum() < 50:
            continue
        features_list.append(feats)
        labels_list.append(labels)
        dates_list.append(mdate)
    print(f'  {len(dates_list)} snapshots ({time.time() - t0:.1f}s)')

    # 4 ── 训练 (使用模块) ──
    print('[4/6] Training GAT...')
    t0 = time.time()
    n_train = int(len(dates_list) * 0.6)
    model = StockGAT().to(DEVICE)
    ei = edge_index.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = GraphUtils.ranking_loss

    trainer = GraphTrainer(model, epochs=args.epochs, patience=20, verbose=True)
    trainer.fit_batches(
        features_list, labels_list, ei,
        n_train=n_train, optimizer=opt, loss_fn=loss_fn, device=DEVICE,
    )

    # 5 ── 保存 ──
    model_path = BASE / 'gnn_model.pt'
    cpu_w = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
    torch.save({
        'model_state_dict': cpu_w, 'features': TOP_FACTORS,
        'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
        'stock_codes': stock_codes, 'edge_index': edge_index.cpu(),
        'model_type': 'GAT', 'gat_heads': GAT_HEADS, 'k_neighbors': K_NEIGHBORS,
    }, model_path)
    with open(BASE / 'gnn_config.json', 'w') as f:
        json.dump({
            'features': TOP_FACTORS, 'hidden_dim': HIDDEN_DIM, 'dropout': DROPOUT,
            'n_stocks': n_stocks, 'n_train': n_train, 'n_total': len(dates_list),
            'loss': 'pairwise_ranking', 'model': 'GAT', 'heads': GAT_HEADS,
            'k_neighbors': K_NEIGHBORS, 'n_edges': len(edges),
        }, f, indent=2)
    print(f'  Saved: {model_path} ({model_path.stat().st_size / 1024:.0f}KB)')

    # 6 ── 推理 + 评估 ──
    print('[5-6/6] Inference + Evaluation...')
    t0 = time.time()
    model.eval()
    all_preds = {}
    for t in range(len(dates_list)):
        x = torch.from_numpy(features_list[t]).float().to(DEVICE)
        with torch.no_grad():
            preds = model(x, ei).cpu().numpy().ravel()
        for i, code in enumerate(stock_codes):
            if not np.isnan(labels_list[t][i]):
                all_preds[f'{dates_list[t]}|{code}'] = float(preds[i])

    import json as _json
    pred_path = BASE / 'gnn_predictions.json'
    with open(pred_path, 'w') as f:
        _json.dump(all_preds, f)
    print(f'  Saved: {pred_path} ({pred_path.stat().st_size / 1024 / 1024:.0f}MB, {len(all_preds):,} entries)')

    # Top-K precision
    from statistics import mean
    test_start = n_train
    precisions = {5: [], 10: [], 30: []}
    for t in range(test_start, len(dates_list)):
        preds_arr = np.array([all_preds.get(f'{dates_list[t]}|{c}', 0) for c in stock_codes])
        lbls = labels_list[t]
        valid = ~np.isnan(lbls)
        if valid.sum() < 50:
            continue
        true_top30 = set(stock_codes[i] for i in np.argsort(lbls)[-30:])
        for k in [5, 10, 30]:
            pred_top = set(stock_codes[i] for i in np.argsort(preds_arr)[-k:])
            precisions[k].append(len(pred_top & true_top30) / k)

    if any(precisions[k] for k in precisions):
        print('  Top-K Precision (test):')
        for k in [5, 10, 30]:
            if precisions[k]:
                print(f'    Top-{k}: {mean(precisions[k]):.3f}')

    print(f'\nDone: {time.time() - t_total:.0f}s total')


if __name__ == '__main__':
    main()
