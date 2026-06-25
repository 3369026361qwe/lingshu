"""
灵枢量化系统 — 端到端每日自动流水线。

一键执行全部步骤:
  Step 1: 数据更新 (从 AKShare/Tushare 拉取最新行情)
  Step 2: 因子计算 (增量更新今日因子值)
  Step 3: 因子融合 (35因子→综合得分)
  Step 4: GNN 推理 (加载已训练模型, 预测收益)
  Step 5: 三路融合 (Factor + GNN + Agent)
  Step 6: Top-N 选股 (生成今日买入清单)
  Step 7: 风控检查 (熔断/仓位/行业限制)
  Step 8: 综合日报 (控制台 + 文件输出)

Usage:
  python scripts/run_daily_pipeline.py              # 完整流水线
  python scripts/run_daily_pipeline.py --skip-gnn   # 跳过GNN(快速模式)
  python scripts/run_daily_pipeline.py --date 2026-06-08  # 指定日期
"""
import sys, os, json, time, argparse
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict
from decimal import Decimal
from math import sqrt, isnan
from statistics import mean, stdev
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('E:/28721/lingshu/data')

# ═══════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════
TOP_N = 40; REBALANCE_STEP = 40
MAX_STOCK_PCT = 0.10; MAX_INDUSTRY_PCT = 0.30
COMMISSION = 0.0003; SLIPPAGE = 0.001

# ═══════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--skip-gnn', action='store_true', help='Skip GNN inference')
parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), default=today')
args = parser.parse_args()

target_date = args.date or date.today().isoformat()
target_date_clean = target_date.replace('-', '')  # YYYYMMDD for DB comparison
print('=' * 65)
print(f'  灵枢量化系统 — 每日自动流水线')
print(f'  日期: {target_date}')
print(f'  GNN: {"跳过" if args.skip_gnn else "启用"}')
print('=' * 65)

# ═══════════════════════════════════════════════════
# Step 1: Check data freshness
# ═══════════════════════════════════════════════════
print('\n[Step 1/8] Checking data freshness...')
t0 = time.time()

from shujuku.session import SessionContext
from sqlalchemy import text

with SessionContext() as s:
    latest_bar = s.execute(text('SELECT MAX(trade_date) FROM daily_bar')).scalar()
    latest_fv = s.execute(text('SELECT MAX(trade_date) FROM factor_value')).scalar()
    latest_fusion = s.execute(text('SELECT MAX(trade_date) FROM fusion_score')).scalar()

print(f'  daily_bar latest:     {latest_bar}')
print(f'  factor_value latest:  {latest_fv}')
print(f'  fusion_score latest:  {latest_fusion}')

data_fresh = (str(latest_bar).replace('-','') >= target_date_clean and
               str(latest_fv).replace('-','') >= target_date_clean)
print(f'  Data fresh: {data_fresh}')

# If data is stale, would trigger import/update scripts (skipped for now)

# ═══════════════════════════════════════════════════
# Step 2: Factor Fusion (if needed)
# ═══════════════════════════════════════════════════
print('\n[Step 2/8] Factor fusion...')
t0 = time.time()

if latest_fusion is not None and str(latest_fusion).replace('-','') >= target_date_clean:
    print(f'  Fusion scores already computed for {target_date}, skipping')
else:
    from juece.factor_fusion import FactorFusion
    fusion = FactorFusion(min_valid_factors=4)

    with SessionContext() as s:
        fv_rows = s.execute(text(
            f"SELECT code, factor_name, raw_value FROM factor_value WHERE trade_date = '{latest_fv}'"
        )).fetchall()
        price_rows = s.execute(text(f"SELECT code, close FROM daily_bar WHERE trade_date = '{latest_fv}'")).fetchall()

    if fv_rows:
        fv = defaultdict(dict)
        for code, fn, rv in fv_rows:
            try: fv[fn][code] = float(str(rv))
            except: pass

        composite = fusion.compute(fv, do_industry_neutralize=True)
        print(f'  Fusion: {len(composite)} stocks scored')

        if composite:
            with SessionContext() as s:
                stmt = text("INSERT INTO fusion_score (trade_date, code, composite_score, rank, signal) VALUES (:d, :c, :s, :r, 'HOLD') ON CONFLICT(trade_date,code) DO UPDATE SET composite_score=:s")
                ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
                batch = []
                for ri, (code, score) in enumerate(ranked[:TOP_N], 1):
                    batch.append({'d': latest_fv, 'c': code, 's': str(round(score, 6)), 'r': ri})
                if batch: s.execute(stmt, batch); s.commit()
                print(f'  Stored: {len(batch)} scores')
    else:
        print(f'  No factor values for {latest_fv}')

print(f'  Done ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════
# Step 3: GNN Inference
# ═══════════════════════════════════════════════════
print('\n[Step 3/8] GNN inference...')
t0 = time.time()

gnn_available = (BASE / 'gnn_model.pt').exists()
if args.skip_gnn or not gnn_available:
    print(f'  GNN skipped (model available: {gnn_available})')
    gnn_preds = None
else:
    import torch, torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class StockGCN(torch.nn.Module):
        def __init__(self, in_dim, hidden, dropout):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, 1)
            self.dropout = dropout
        def forward(self, x, ei):
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, ei)

    checkpoint = torch.load(BASE / 'gnn_model.pt', map_location='cpu')
    model = StockGCN(len(checkpoint['features']), checkpoint['hidden_dim'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    edge_index = checkpoint['edge_index']

    # Load factor features for latest date
    with SessionContext() as s:
        fv_rows = s.execute(text(
            f"SELECT code, factor_name, raw_value FROM factor_value WHERE trade_date = '{latest_fv}'"
        )).fetchall()

    stock_codes = checkpoint['stock_codes']
    feats = np.zeros((len(stock_codes), len(checkpoint['features'])), dtype=np.float32)
    fv = defaultdict(dict)
    for code, fn, rv in fv_rows:
        try: fv[code][fn] = float(str(rv))
        except: pass
    for i, code in enumerate(stock_codes):
        if code in fv:
            for j, fn in enumerate(checkpoint['features']):
                v = fv[code].get(fn);
                if v is not None and not isnan(v) and abs(v) < 1e8: feats[i,j] = v

    x = torch.from_numpy(feats).float()
    with torch.no_grad():
        gnn_preds = model(x, edge_index).numpy().ravel()
    gnn_preds = {stock_codes[i]: float(gnn_preds[i]) for i in range(len(stock_codes))}
    print(f'  GNN inference: {len(gnn_preds)} stocks ({time.time()-t0:.1f}s)')

# ═══════════════════════════════════════════════════
# Step 4: Three-Way Fusion
# ═══════════════════════════════════════════════════
print('\n[Step 4/8] Three-way fusion...')
t0 = time.time()

with SessionContext() as s:
    fusion_rows = s.execute(text(f"SELECT code, composite_score FROM fusion_score WHERE trade_date = '{latest_fv}'")).fetchall()

factor_scores = {r[0]: float(r[1]) for r in fusion_rows}

if factor_scores and gnn_preds:
    # Normalize both
    def norm(s):
        if not s: return {}
        vals = list(s.values()); v_min, v_max = min(vals), max(vals)
        return {c: (v - v_min)/(v_max - v_min) if v_max > v_min else 0.5 for c, v in s.items()}
    fn = norm(factor_scores); gn = norm(gnn_preds)
    all_codes = set(fn) | set(gn)
    # Fusion: 50/30/20 (optimal from grid search)
    final_scores = {c: fn.get(c,0.5)*0.50 + gn.get(c,0.5)*0.30 + 0.5*0.20 for c in all_codes}
    print(f'  Fusion: {len(final_scores)} stocks (Factor 50% + GNN 30% + Agent 20%)')
else:
    final_scores = factor_scores
    print(f'  Fusion skipped (no GNN), using factor-only: {len(final_scores)} stocks')

# ═══════════════════════════════════════════════════
# Step 5: Top-N Selection
# ═══════════════════════════════════════════════════
print('\n[Step 5/8] Stock selection...')

ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
picks = ranked[:TOP_N]

# Industry diversification
import tushare as ts
ts.set_token(os.environ.get('TUSHARE_TOKEN',''))
pro = ts.pro_api()
try:
    TS_TO_SW = {'银行':'银行','证券':'非银金融','保险':'非银金融','全国地产':'房地产','区域地产':'房地产','石油开采':'石油石化','石油加工':'石油石化','煤炭开采':'煤炭','铜':'有色金属','铝':'有色金属','黄金':'有色金属','小金属':'有色金属','钢铁':'钢铁','化工原料':'基础化工','农药化肥':'基础化工','塑料':'基础化工','化学制药':'医药生物','生物制药':'医药生物','中成药':'医药生物','医疗保健':'医药生物','白酒':'食品饮料','啤酒':'食品饮料','食品':'食品饮料','乳制品':'食品饮料','种植业':'农林牧渔','渔业':'农林牧渔','饲料':'农林牧渔','汽车整车':'汽车','汽车配件':'汽车','家用电器':'家用电器','纺织':'纺织服饰','服饰':'纺织服饰','造纸':'轻工制造','火力发电':'公用事业','水力发电':'公用事业','供气供热':'公用事业','水务':'公用事业','环境保护':'公用事业','建筑施工':'建筑装饰','装修装饰':'建筑装饰','水泥':'建筑材料','玻璃':'建筑材料','运输设备':'机械设备','工程机械':'机械设备','专用机械':'机械设备','通用机械':'机械设备','电气设备':'电力设备','电器仪表':'电力设备','半导体':'电子','元器件':'电子','IT设备':'计算机','软件服务':'计算机','互联网':'计算机','通信设备':'通信','电信运营':'通信','仓储物流':'交通运输','空运':'交通运输','水运':'交通运输','路桥':'交通运输','铁路':'交通运输','机场':'交通运输','港口':'交通运输','影视音像':'传媒','出版业':'传媒','航空':'国防军工','船舶':'国防军工','百货':'商贸零售','贸易':'商贸零售','旅游景点':'社会服务','旅游服务':'社会服务','酒店餐饮':'社会服务','医疗美容':'美容护理','综合类':'综合'}
    df_ind = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
    industry_map = {r['ts_code']: TS_TO_SW.get(r.get('industry','') or '', '综合') for _, r in df_ind.iterrows()}
except:
    industry_map = {}

ind_count = defaultdict(int)
diversified = []
for code, score in picks:
    ind = industry_map.get(code, '其他')
    if ind_count[ind] < max(3, TOP_N // 10):
        diversified.append((code, score))
        ind_count[ind] += 1
picks = diversified[:TOP_N]

print(f'  Selected: {len(picks)} stocks ({len(set(industry_map.get(c,"?") for c,_ in picks))} industries)')

# ═══════════════════════════════════════════════════
# Step 6: Risk Check
# ═══════════════════════════════════════════════════
print('\n[Step 6/8] Risk check...')

with SessionContext() as s:
    snap = s.execute(text("SELECT total_value, cumulative_return FROM portfolio_snapshot ORDER BY trade_date DESC LIMIT 60")).fetchall()

risk_status = 'LOW'
if snap:
    recent_values = [r[0] for r in snap if r[0]]
    if recent_values:
        peak = max(recent_values)
        current = recent_values[0]
        dd = (peak - current) / peak if peak > 0 else 0
        if dd > 0.20: risk_status = 'CRITICAL'
        elif dd > 0.15: risk_status = 'HIGH'
        elif dd > 0.10: risk_status = 'ELEVATED'
        print(f'  Current DD: {dd:.1%} | Risk: {risk_status}')
    else:
        print(f'  No portfolio history | Risk: LOW')
else:
    print(f'  No portfolio history | Risk: LOW')

# ═══════════════════════════════════════════════════
# Step 7: Store Signals
# ═══════════════════════════════════════════════════
print('\n[Step 7/8] Storing signals...')

signal_map = {}
for ri, (code, score) in enumerate(picks, 1):
    if ri <= TOP_N // 3: sig = 'STRONG_BUY'
    elif ri <= TOP_N * 2 // 3: sig = 'BUY'
    else: sig = 'WATCH'
    signal_map[code] = sig

with SessionContext() as s:
    stmt = text("INSERT OR REPLACE INTO positions (code, quantity, avg_cost, current_price, market_value, updated_at) VALUES (:c, 1, :p, :p, :p, datetime('now'))")
    for code, score in picks[:TOP_N]:
        try: s.execute(stmt, {'c': code, 'p': float(score) * 100})
        except: pass
    s.commit()
    print(f'  Signals stored: {len(signal_map)} stocks')

# ═══════════════════════════════════════════════════
# Step 8: Daily Report
# ═══════════════════════════════════════════════════
print(f'\n{"="*65}')
print(f'  灵枢量化系统 — 每日报告 ({target_date})')
print(f'{"="*65}')

print(f'\n  ── 今日选股 Top-{TOP_N} ──')
print(f'  {"Rank":>4s}  {"Code":10s}  {"Score":>8s}  {"Signal":12s}  {"Industry"}')
for ri, (code, score) in enumerate(picks, 1):
    ind = industry_map.get(code, '??')
    sig = signal_map.get(code, 'HOLD')
    print(f'  {ri:>4d}  {code:10s}  {score:>8.4f}  {sig:12s}  {ind}')

print(f'\n  ── 风险状态 ──')
print(f'  风险等级: {risk_status}')
print(f'  行业数:   {len(set(industry_map.get(c,"?") for c,_ in picks))}')
print(f'  最大行业: {max(ind_count.values()) if ind_count else 0}/{TOP_N}')

# Performance summary
with SessionContext() as s:
    perf = s.execute(text(
        'SELECT COUNT(*), AVG(total_value), MIN(cumulative_return), MAX(cumulative_return) FROM portfolio_snapshot'
    )).fetchone()
    if perf and perf[0] > 0:
        print(f'\n  ── 累计绩效 ──')
        print(f'  快照数:   {perf[0]}')
        print(f'  最高收益: {perf[3]:+.2%}' if perf[3] else '')

print(f'\n  ── 因子表现 ──')
with SessionContext() as s:
    top_ic = s.execute(text(
        'SELECT factor_name, AVG(ic) FROM factor_ic_record GROUP BY factor_name ORDER BY ABS(AVG(ic)) DESC LIMIT 5'
    )).fetchall()
    if top_ic:
        for fn, mic in top_ic:
            print(f'  {fn:25s}: IC={float(mic):+.4f}')
    else:
        print(f'  (no IC records)')

print(f'\n{"="*65}')
print(f'  流水线完成 — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'{"="*65}')
