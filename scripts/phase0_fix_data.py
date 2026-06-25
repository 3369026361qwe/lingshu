"""Phase 0: 数据修复 - 行业分类 + z_score回填 + IR回填"""
import sys, os, time
from pathlib import Path
from statistics import mean, stdev
from math import isnan
from dotenv import load_dotenv
load_dotenv(Path('E:/28721/lingshu/.env'))
sys.stdout.reconfigure(encoding='utf-8')

from shujuku.session import SessionContext
from sqlalchemy import text

TS_TO_SW = {'银行':'银行','证券':'非银金融','保险':'非银金融','多元金融':'非银金融','全国地产':'房地产','区域地产':'房地产','房产服务':'房地产','石油开采':'石油石化','石油加工':'石油石化','石油贸易':'石油石化','煤炭开采':'煤炭','焦炭加工':'煤炭','铜':'有色金属','铝':'有色金属','铅锌':'有色金属','黄金':'有色金属','小金属':'有色金属','矿物制品':'有色金属','钢铁':'钢铁','特种钢':'钢铁','普钢':'钢铁','化工原料':'基础化工','农药化肥':'基础化工','塑料':'基础化工','橡胶':'基础化工','染料涂料':'基础化工','化纤':'基础化工','日用化工':'基础化工','化学制药':'医药生物','生物制药':'医药生物','中成药':'医药生物','医药商业':'医药生物','医疗保健':'医药生物','白酒':'食品饮料','啤酒':'食品饮料','软饮料':'食品饮料','食品':'食品饮料','乳制品':'食品饮料','调味品':'食品饮料','种植业':'农林牧渔','渔业':'农林牧渔','饲料':'农林牧渔','农业综合':'农林牧渔','汽车整车':'汽车','汽车配件':'汽车','摩托车':'汽车','家用电器':'家用电器','家居用品':'家用电器','纺织':'纺织服饰','服饰':'纺织服饰','纺织机械':'机械设备','造纸':'轻工制造','包装印刷':'轻工制造','文教休闲':'轻工制造','火力发电':'公用事业','水力发电':'公用事业','新型电力':'公用事业','供气供热':'公用事业','水务':'公用事业','环境保护':'公用事业','建筑施工':'建筑装饰','装修装饰':'建筑装饰','建筑工程':'建筑装饰','水泥':'建筑材料','玻璃':'建筑材料','陶瓷':'建筑材料','其他建材':'建筑材料','运输设备':'机械设备','工程机械':'机械设备','农用机械':'机械设备','机械基件':'机械设备','机床制造':'机械设备','专用机械':'机械设备','通用机械':'机械设备','轻工机械':'机械设备','电气设备':'电力设备','电器仪表':'电力设备','半导体':'电子','元器件':'电子','电子制造':'电子','IT设备':'计算机','软件服务':'计算机','互联网':'计算机','电脑设备':'计算机','通信设备':'通信','电信运营':'通信','仓储物流':'交通运输','空运':'交通运输','水运':'交通运输','路桥':'交通运输','铁路':'交通运输','机场':'交通运输','港口':'交通运输','公共交通':'交通运输','影视音像':'传媒','出版业':'传媒','广告代理':'传媒','航空':'国防军工','船舶':'国防军工','军工机械':'国防军工','百货':'商贸零售','超市连锁':'商贸零售','贸易':'商贸零售','其他商业':'商贸零售','商品城':'商贸零售','旅游景点':'社会服务','旅游服务':'社会服务','酒店餐饮':'社会服务','医疗美容':'美容护理','综合类':'综合','园区开发':'房地产'}

print("=" * 55)
print("  Phase 0: Data Fixes")
print("=" * 55)

# Fix 1: industry_classification
print("\n[1/3] industry_classification...")
t0 = time.time()
import tushare as ts
ts.set_token(os.environ['TUSHARE_TOKEN'])
pro = ts.pro_api()
df_ind = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')

with SessionContext() as s:
    stmt = text("INSERT OR REPLACE INTO industry_classification (code, sw_level1, effective_date, source, updated_at) VALUES (:c, :sw, '2026-01-01', 'Tushare', datetime('now'))")
    batch = [{'c': r['ts_code'], 'sw': TS_TO_SW.get(r.get('industry','') or '','综合')} for _, r in df_ind.iterrows()]
    s.execute(stmt, batch); s.commit()
    cnt = s.execute(text('SELECT COUNT(*) FROM industry_classification')).scalar()
print(f"  Done: {cnt:,} rows ({time.time()-t0:.1f}s)")

# Fix 2: z_score + percentile
print("\n[2/3] factor_value z_score + percentile...")
t0 = time.time()

with SessionContext() as s:
    dates = [r[0] for r in s.execute(text('SELECT DISTINCT trade_date FROM factor_value ORDER BY trade_date')).fetchall()]
    factors = [r[0] for r in s.execute(text('SELECT DISTINCT factor_name FROM factor_value ORDER BY factor_name')).fetchall()]
print(f"  {len(dates)} dates x {len(factors)} factors")

updated = 0
for di, td in enumerate(dates):
    td_str = str(td)
    for fn in factors:
        with SessionContext() as s:
            rows = s.execute(text(
                "SELECT id, raw_value FROM factor_value WHERE trade_date=:d AND factor_name=:f"
            ), {'d': td_str, 'f': fn}).fetchall()
            if len(rows) < 30: continue
            vals = []
            for rid, rv in rows:
                try:
                    v = float(str(rv))
                    if not isnan(v) and abs(v) < 1e8: vals.append((rid, v))
                except: pass
            if len(vals) < 30: continue
            vs = [v for _, v in vals]
            mu = mean(vs); sg = stdev(vs) if len(vs) > 1 else 1.0
            if sg < 1e-12: sg = 1.0
            sorted_vs = sorted(enumerate(vs), key=lambda x: x[1])
            n = len(sorted_vs)
            rank_map = {i: (rk+1)/n for rk, (i, _) in enumerate(sorted_vs)}
            stmt = text("UPDATE factor_value SET z_score=:z, percentile=:p WHERE id=:id")
            for idx, (rid, v) in enumerate(vals):
                s.execute(stmt, {'id': rid, 'z': round((v-mu)/sg,6), 'p': round(rank_map[idx],6)})
            s.commit()
            updated += len(vals)
    if (di+1) % 100 == 0:
        print(f"  {di+1}/{len(dates)} dates ({updated:,} updates)")

with SessionContext() as s:
    zc = s.execute(text('SELECT COUNT(*) FROM factor_value WHERE z_score IS NOT NULL')).scalar()
print(f"  Done: z_score filled for {zc:,} rows ({time.time()-t0:.0f}s)")

# Fix 3: IR backfill
print("\n[3/3] factor_ic_record IR...")
t0 = time.time()
with SessionContext() as s:
    fn_list = [r[0] for r in s.execute(text('SELECT DISTINCT factor_name FROM factor_ic_record')).fetchall()]
    for fn in fn_list:
        rows = s.execute(text("SELECT id, ic FROM factor_ic_record WHERE factor_name=:f ORDER BY trade_date"), {'f': fn}).fetchall()
        if len(rows) < 12: continue
        ics = [float(r[1]) for r in rows]
        mic = mean(ics); sic = stdev(ics) if len(ics) > 1 else 0.01
        ir = round(mic/sic, 6) if sic > 0 else 0
        s.execute(text("UPDATE factor_ic_record SET ir=:ir WHERE factor_name=:f"), {'ir': ir, 'f': fn})
    s.commit()
    cnt = s.execute(text('SELECT COUNT(*) FROM factor_ic_record WHERE ir IS NOT NULL')).scalar()
print(f"  Done: {cnt:,} IR values ({time.time()-t0:.1f}s)")
print(f"\nPhase 0 Complete")
