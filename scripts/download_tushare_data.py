# ruff: noqa: E999 (datetime("now") in SQL strings confuses parser)
"""
Tushare 全量数据下载 — 股票列表 → 日线行情 → 财务数据 → 行业分类。
直接灌入 SQLite 数据库，拉完即可运行因子计算 + 每日流水线。

Usage:
    python scripts/download_tushare_data.py                    # 全部下载
    python scripts/download_tushare_data.py --quick            # 快速模式 (仅股票列表+近6月日线)
    python scripts/download_tushare_data.py --stocks-only      # 仅股票列表
    python scripts/download_tushare_data.py --skip-financials  # 跳过财务数据
"""
import sys, os, time, argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from decimal import Decimal

# 确保项目根目录在 path

# Windows 控制台 UTF-8 编码
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

import tushare as ts

TOKEN = os.environ.get('TUSHARE_TOKEN', '')
if not TOKEN:
    print('ERROR: TUSHARE_TOKEN not set in .env')
    sys.exit(1)

ts.set_token(TOKEN)
pro = ts.pro_api()

from shujuku.session import init_db, SessionContext
from sqlalchemy import text

# ── 参数 ──────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Tushare 全量数据下载')
parser.add_argument('--quick', action='store_true', help='快速模式：仅近 6 个月日线')
parser.add_argument('--stocks-only', action='store_true', help='仅下载股票列表')
parser.add_argument('--skip-financials', action='store_true', help='跳过财务数据')
args = parser.parse_args()

# ── 初始化 ────────────────────────────────────────────

init_db(drop_all=False)
print('[OK] 数据库表已就绪\n')

# 申万行业映射 (Tushare行业 → 申万一级)
TS_TO_SW = {
    '银行': '银行', '证券': '非银金融', '保险': '非银金融',
    '全国地产': '房地产', '区域地产': '房地产',
    '石油开采': '石油石化', '石油加工': '石油石化',
    '煤炭开采': '煤炭', '铜': '有色金属', '铝': '有色金属', '黄金': '有色金属', '小金属': '有色金属',
    '钢铁': '钢铁', '化工原料': '基础化工', '农药化肥': '基础化工', '塑料': '基础化工',
    '化学制药': '医药生物', '生物制药': '医药生物', '中成药': '医药生物', '医疗保健': '医药生物',
    '白酒': '食品饮料', '啤酒': '食品饮料', '食品': '食品饮料', '乳制品': '食品饮料',
    '种植业': '农林牧渔', '渔业': '农林牧渔', '饲料': '农林牧渔',
    '汽车整车': '汽车', '汽车配件': '汽车', '家用电器': '家用电器',
    '纺织': '纺织服饰', '服饰': '纺织服饰', '造纸': '轻工制造',
    '火力发电': '公用事业', '水力发电': '公用事业', '供气供热': '公用事业', '水务': '公用事业', '环境保护': '公用事业',
    '建筑施工': '建筑装饰', '装修装饰': '建筑装饰', '水泥': '建筑材料', '玻璃': '建筑材料',
    '运输设备': '机械设备', '工程机械': '机械设备', '专用机械': '机械设备', '通用机械': '机械设备',
    '电气设备': '电力设备', '电器仪表': '电力设备',
    '半导体': '电子', '元器件': '电子',
    'IT设备': '计算机', '软件服务': '计算机', '互联网': '计算机',
    '通信设备': '通信', '电信运营': '通信',
    '仓储物流': '交通运输', '空运': '交通运输', '水运': '交通运输', '路桥': '交通运输',
    '铁路': '交通运输', '机场': '交通运输', '港口': '交通运输',
    '影视音像': '传媒', '出版业': '传媒',
    '航空': '国防军工', '船舶': '国防军工',
    '百货': '商贸零售', '贸易': '商贸零售',
    '旅游景点': '社会服务', '旅游服务': '社会服务', '酒店餐饮': '社会服务',
    '医疗美容': '美容护理', '综合类': '综合',
}

# ═══════════════════════════════════════════════════════
# Step 1: 股票列表
# ═══════════════════════════════════════════════════════

print('=' * 60)
print('[1/4] 下载股票列表 + 行业分类...')
t0 = time.time()

df_stocks = pro.stock_basic(exchange='', list_status='L',
                            fields='ts_code,symbol,name,area,industry,list_date')
print(f'  Tushare 返回: {len(df_stocks)} 只')

# 过滤: 仅保留沪深两市 (排除 BJ 北交所)
df_stocks = df_stocks[df_stocks['ts_code'].str.contains('.SH|.SZ')]
print(f'  沪深两市: {len(df_stocks)} 只')

_BATCH = 500
si_batch, ind_batch = [], []

for _, row in df_stocks.iterrows():
    code = row["ts_code"]
    exchange = "SH" if ".SH" in code else "SZ"
    name = str(row.get("name", "") or "")
    list_date = row.get("list_date", "")
    if list_date and str(list_date) != "nan":
        ld = date(int(str(list_date)[:4]), int(str(list_date)[4:6]), int(str(list_date)[6:8]))
    else:
        ld = None
    si_batch.append({"c": code, "n": name, "x": exchange, "d": ld})
    raw_ind = str(row.get("industry", "") or "")
    sw1 = TS_TO_SW.get(raw_ind, raw_ind if raw_ind else "综合")
    if sw1:
        ind_batch.append({"c": code, "sw": sw1})

for i in range(0, len(si_batch), _BATCH):
    chunk = si_batch[i:i + _BATCH]
    with SessionContext() as s:
        for b in chunk:
            s.execute(text(
                "INSERT INTO stock_info (code, name, exchange, listing_date, is_active, updated_at) "
                "VALUES (:c, :n, :x, :d, 1, datetime("now")) "
                "ON CONFLICT(code) DO UPDATE SET name=:n, is_active=1, updated_at=datetime("now")"
            ), b)
        s.commit()

for i in range(0, len(ind_batch), _BATCH):
    chunk = ind_batch[i:i + _BATCH]
    with SessionContext() as s:
        for b in chunk:
            s.execute(text(
                "INSERT INTO industry_classification (code, sw_level1, effective_date, source, updated_at) "
                "VALUES (:c, :sw, date("now"), "Tushare", datetime("now"))"
            ), b)
        s.commit()

print(f"  StockInfo: {len(si_batch)} | Industry: {len(ind_batch)}")

    result_si = s.execute(text('SELECT COUNT(*) FROM stock_info')).scalar()
    result_ind = s.execute(text('SELECT COUNT(*) FROM industry_classification')).scalar()

print(f'  [OK] stock_info: {result_si} | industry_classification: {result_ind}')
print(f'  耗时: {time.time()-t0:.0f}s\n')

if args.stocks_only:
    print('=== --stocks-only, 退出 ===')
    sys.exit(0)

# ═══════════════════════════════════════════════════════
# Step 2: 日线行情
# ═══════════════════════════════════════════════════════

print('[2/4] 下载日线行情...')
t0 = time.time()

if args.quick:
    start_date = (date.today() - timedelta(days=180)).strftime('%Y%m%d')
else:
    start_date = '20240101'
end_date = date.today().strftime('%Y%m%d')
print(f'  日期范围: {start_date} ~ {end_date}')

codes = df_stocks['ts_code'].tolist()
print(f'  待下载: {len(codes)} 只')

BATCH = 500
bar_count, batch_num = 0, 0

with SessionContext() as s:
    for i in range(0, len(codes), 50):
        batch_codes = codes[i:i + 50]
        code_str = ','.join(batch_codes)

        try:
            df_bars = pro.daily(
                ts_code=code_str,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,vol,amount,turnover_rate'
            )
            time.sleep(0.3)  # Tushare 免费版限频

            if df_bars is None or len(df_bars) == 0:
                continue

            batch_vals = []
            for _, bar in df_bars.iterrows():
                batch_vals.append({
                    'c': bar['ts_code'],
                    'd': bar['trade_date'],
                    'o': str(bar.get('open', 0) or 0),
                    'h': str(bar.get('high', 0) or 0),
                    'l': str(bar.get('low', 0) or 0),
                    'cl': str(bar.get('close', 0) or 0),
                    'v': str(bar.get('vol', 0) or 0),
                    'a': str(bar.get('amount', 0) or 0),
                    't': str(bar.get('turnover_rate', 0) or 0),
                })

            if batch_vals:
                s.execute(text(
                    "INSERT INTO daily_bar (code, trade_date, open, high, low, close, volume, amount, turnover_rate, is_st, updated_at) "
                    "VALUES (:c, :d, :o, :h, :l, :cl, :v, :a, :t, 0, datetime('now')) "
                    "ON CONFLICT(code, trade_date) DO NOTHING"
                ), batch_vals)
                s.commit()
                bar_count += len(batch_vals)

            batch_num += 1
            if batch_num % 20 == 0:
                elapsed = time.time() - t0
                result_db = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()
                print(f'  进度: {i+len(batch_codes):>4d}/{len(codes)} 只 | DB累计: {result_db:>8,} 行 | {elapsed:.0f}s')

        except Exception as e:
            print(f'  [WARN] {code_str[:20]}... 失败: {e}')
            time.sleep(1)
            continue

    result_db = s.execute(text('SELECT COUNT(*) FROM daily_bar')).scalar()

print(f'  [OK] daily_bar: {result_db:,} 行')
print(f'  耗时: {time.time()-t0:.0f}s\n')

# ═══════════════════════════════════════════════════════
# Step 3: 财务数据
# ═══════════════════════════════════════════════════════

if args.skip_financials:
    print('[3/4] 财务数据 — 跳过\n')
else:
    print('[3/4] 下载财务数据 (最近一期)...')
    t0 = time.time()

    fin_count = 0
    with SessionContext() as s:
        for i in range(0, len(codes), 30):
            batch_codes = codes[i:i + 30]
            code_str = ','.join(batch_codes)

            try:
                # 最新一期财报指标
                df_fin = pro.fina_indicator(ts_code=code_str, period='20260331',
                                           fields='ts_code,end_date,pe,pb,ps,roe,roa,grossprofit_margin,netprofit_margin,revenue,np,ocf')
                time.sleep(0.5)

                if df_fin is None or len(df_fin) == 0:
                    continue

                batch_vals = []
                for _, row in df_fin.iterrows():
                    end_d = str(row.get('end_date', '') or '')
                    batch_vals.append({
                        'c': row['ts_code'],
                        'd': end_d,
                        't': 'Q1',
                        'pe': str(row.get('pe', '') or ''),
                        'pb': str(row.get('pb', '') or ''),
                        'ps': str(row.get('ps', '') or ''),
                        'roe': str(row.get('roe', '') or ''),
                        'roa': str(row.get('roa', '') or ''),
                        'gm': str(row.get('grossprofit_margin', '') or ''),
                        'nm': str(row.get('netprofit_margin', '') or ''),
                        'rev': str(row.get('revenue', '') or ''),
                        'np': str(row.get('np', '') or ''),
                        'ocf': str(row.get('ocf', '') or ''),
                    })

                if batch_vals:
                    s.execute(text(
                        "INSERT INTO financial_report (code, report_date, report_type, pe, pb, ps, roe, roa, gross_margin, net_margin, revenue, net_profit, operating_cashflow, updated_at) "
                        "VALUES (:c, :d, :t, :pe, :pb, :ps, :roe, :roa, :gm, :nm, :rev, :np, :ocf, datetime('now')) "
                        "ON CONFLICT(code, report_date) DO UPDATE SET pe=:pe, pb=:pb, ps=:ps, roe=:roe, roa=:roa, gross_margin=:gm, net_margin=:nm, revenue=:rev, net_profit=:np, operating_cashflow=:ocf, updated_at=datetime('now')"
                    ), batch_vals)
                    s.commit()
                    fin_count += len(batch_vals)

            except Exception as e:
                if '每分钟最多访问' in str(e):
                    print(f'  [WARN] 频率限制，等待 60s...')
                    time.sleep(60)
                    continue
                continue

        result_fin = s.execute(text('SELECT COUNT(*) FROM financial_report')).scalar()

    print(f'  [OK] financial_report: {result_fin} 行')
    print(f'  耗时: {time.time()-t0:.0f}s\n')

# ═══════════════════════════════════════════════════════
# Step 4: 验证
# ═══════════════════════════════════════════════════════

print('[4/4] 数据验证...')

with SessionContext() as s:
    tables = {
        'stock_info': 'SELECT COUNT(*) FROM stock_info',
        'industry_classification': 'SELECT COUNT(*) FROM industry_classification',
        'daily_bar': 'SELECT COUNT(*) FROM daily_bar',
        'financial_report': 'SELECT COUNT(*) FROM financial_report',
    }
    for name, sql in tables.items():
        cnt = s.execute(text(sql)).scalar()
        print(f'  {name:30s}: {cnt:>8,} 行')

    # 检查最新数据日期
    latest = s.execute(text('SELECT MAX(trade_date) FROM daily_bar')).scalar()
    print(f'\n  最新日线日期: {latest}')

    # 样本查询
    sample = s.execute(text(
        "SELECT code, trade_date, close FROM daily_bar WHERE code='000001.SZ' ORDER BY trade_date DESC LIMIT 3"
    )).fetchall()
    print(f'\n  样本 (000001.SZ):')
    for r in sample:
        print(f'    {r[0]}  {r[1]}  close={r[2]}')

db_path = Path(__file__).resolve().parent.parent / 'data' / 'lingshu.db'
if db_path.exists():
    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f'\n  [DB] 数据库文件: {size_mb:.0f} MB')

print('\n' + '=' * 60)
print('[OK] 数据下载完成！')
print('  下一步: python scripts/compute_factors_full.py')
print('=' * 60)
