"""
简化回测: fusion_score → portfolio_snapshot + risk_logs + var_records。
铁律: INSERT ON CONFLICT DO NOTHING，保护已有数据。

Strategy: 等权持有 Top 20 选股，40 天调仓一次。
"""
import sys, os, time
from pathlib import Path
from collections import defaultdict
from math import sqrt, isnan
from statistics import mean, stdev
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from shujuku.session import SessionContext
from sqlalchemy import text
import numpy as np

TOP_N = 20
REBAL_FREQ = 40  # 交易日
INITIAL_CAPITAL = 1_000_000.0

print('=' * 60)
print(f'简化回测 | Top{TOP_N} | 调仓间隔{REBAL_FREQ}天')
print('=' * 60)

with SessionContext() as s:
    # 加载价格数据
    print('\n[1/3] 加载数据...')
    t0 = time.time()

    prices = defaultdict(dict)
    pr_rows = s.execute(text('SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date')).fetchall()
    for code, td, close in pr_rows:
        try: prices[str(td)][code] = float(str(close))
        except: pass

    all_dates = sorted(prices.keys())
    print(f'  价格数据: {len(all_dates)} 天 ({all_dates[0]} ~ {all_dates[-1]})')

    # 加载 fusion_score
    fusion = defaultdict(dict)
    fs_rows = s.execute(text('SELECT trade_date, code, composite_score, rank FROM fusion_score ORDER BY trade_date, rank')).fetchall()
    for td, code, sc, rk in fs_rows:
        fusion[str(td)][code] = float(sc)

    print(f'  fusion_score: {len(fusion)} 天')
    print(f'  加载耗时: {time.time()-t0:.0f}s')

    # ── 回测 ──────────────────────────────────────────

    print('\n[2/3] 回测模拟...')
    t0 = time.time()

    cash = INITIAL_CAPITAL
    holdings = {}  # {code: quantity}
    snapshots = []
    daily_rets = []
    risk_events = []
    var_values = []

    peak_value = INITIAL_CAPITAL
    rebal_day = 0

    for di, trade_date in enumerate(all_dates):
        # 获取当日价格
        day_prices = prices.get(trade_date, {})
        if not day_prices:
            continue

        # 调仓日
        rebal_day += 1
        if rebal_day >= REBAL_FREQ or not holdings:
            rebal_day = 0

            # 卖出全部
            for code, qty in list(holdings.items()):
                px = day_prices.get(code)
                if px and qty > 0:
                    cash += qty * px
                del holdings[code]

            # 买入 Top N
            picks = fusion.get(trade_date, {})
            ranked = sorted(picks.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
            if ranked:
                weight_per_stock = cash / TOP_N
                for code, score in ranked:
                    px = day_prices.get(code)
                    if px and px > 0:
                        qty = int(weight_per_stock / px / 100) * 100  # 整手
                        if qty > 0:
                            holdings[code] = qty
                            cash -= qty * px

        # 计算当日市值
        market_value = cash
        for code, qty in holdings.items():
            px = day_prices.get(code)
            if px:
                market_value += qty * px

        total_value = market_value

        # 日收益率
        if len(snapshots) > 0:
            prev_val = float(snapshots[-1]['tv'])
            daily_ret = (total_value - prev_val) / prev_val if prev_val > 0 else 0
        else:
            daily_ret = 0

        daily_rets.append(daily_ret)

        # 回撤检测
        if total_value > peak_value:
            peak_value = total_value
        drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0

        level = 'LOW'
        if drawdown > 0.20: level = 'CRITICAL'
        elif drawdown > 0.15: level = 'HIGH'
        elif drawdown > 0.10: level = 'MEDIUM'

        if level in ('CRITICAL', 'HIGH'):
            risk_events.append({
                'd': trade_date, 'dd': round(drawdown * 100, 1), 'lvl': level,
                'scale': round(1.0 - drawdown, 2),
            })

        # 快照
        cum_ret = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
        snapshots.append({
            'd': trade_date, 'tv': str(round(total_value, 2)),
            'cash': str(round(cash, 2)), 'mv': str(round(total_value - cash, 2)),
            'dr': str(round(daily_ret, 8)), 'cr': str(round(cum_ret, 8)),
            'pc': len(holdings),
        })

    # ── VaR 计算 (绝对金额) ─────────────────────────

    if len(daily_rets) >= 20:
        for i in range(20, len(daily_rets)):
            window = daily_rets[i-20:i]
            mu = mean(window)
            sigma = stdev(window) if len(window) > 1 else 0.01
            var_95_pct = mu - 1.645 * sigma
            cvar_vals = [r for r in window if r <= var_95_pct]
            cvar_95_pct = mean(cvar_vals) if cvar_vals else var_95_pct

            # 转换为绝对金额 (基于当日总资产)
            curr_val = float(snapshots[i]['tv']) if i < len(snapshots) else INITIAL_CAPITAL
            var_abs = round(var_95_pct * curr_val, 2)
            cvar_abs = round(cvar_95_pct * curr_val, 2)

            var_values.append({
                'd': all_dates[i] + 'T15:00:00',  # DateTime format
                'var95': str(var_abs),
                'cvar95': str(cvar_abs),
            })

    elapsed = time.time() - t0
    print(f'  快照: {len(snapshots)} | 风控事件: {len(risk_events)} | VaR: {len(var_values)}')
    print(f'  回测耗时: {elapsed:.0f}s')

    # ── 写入 ──────────────────────────────────────────

    print('\n[3/3] 写入数据库...')

    snap_before = s.execute(text('SELECT COUNT(*) FROM portfolio_snapshot')).scalar()
    risk_before = s.execute(text('SELECT COUNT(*) FROM risk_logs')).scalar()
    var_before = s.execute(text('SELECT COUNT(*) FROM var_records')).scalar()

    # Portfolio snapshots
    if snapshots:
        stmt = text(
            'INSERT INTO portfolio_snapshot (trade_date, total_value, cash, market_value, daily_return, cumulative_return, position_count, updated_at) '
            'VALUES (:d, :tv, :cash, :mv, :dr, :cr, :pc, datetime("now")) '
            'ON CONFLICT(trade_date) DO NOTHING'
        )
        for i in range(0, len(snapshots), 2000):
            s.execute(stmt, snapshots[i:i+2000])
        s.commit()

    # Risk logs — 先检查是否存在，防重复
    if risk_events:
        existing_ts = set(r[0] for r in s.execute(text(
            'SELECT DISTINCT timestamp FROM risk_logs'
        )).fetchall())
        new_events = [ev for ev in risk_events if (ev['d'] + 'T15:00:00') not in existing_ts]
        if new_events:
            stmt = text(
                "INSERT INTO risk_logs (timestamp, level, category, message, detail, created_at) "
                "VALUES (:ts, :lvl, 'DRAWDOWN', '回撤触发仓位缩减', :d, datetime('now'))"
            )
            for ev in new_events:
                try:
                    s.execute(stmt, {
                        'ts': ev['d'] + 'T15:00:00',
                        'lvl': 'CRITICAL' if ev['lvl'] == 'CRITICAL' else 'WARNING',
                        'd': 'date=' + ev['d'] + ' drawdown=' + str(ev['dd']) + '%'
                    })
                except Exception:
                    pass
        s.commit()

    # VaR records (no unique constraint, check existence first)
    if var_values:
        existing_dates = set(r[0] for r in s.execute(text(
            'SELECT DISTINCT calc_date FROM var_records'
        )).fetchall())
        new_vars = [v for v in var_values if v['d'] not in existing_dates]
        if new_vars:
            stmt = text(
                "INSERT INTO var_records (calc_date, confidence_level, var, cvar, method, window_days, created_at) "
                "VALUES (:d, 0.95, :var95, :cvar95, 'historical', 20, datetime('now'))"
            )
            for i in range(0, len(new_vars), 2000):
                s.execute(stmt, new_vars[i:i+2000])
            s.commit()

    snap_after = s.execute(text('SELECT COUNT(*) FROM portfolio_snapshot')).scalar()
    risk_after = s.execute(text('SELECT COUNT(*) FROM risk_logs')).scalar()
    var_after = s.execute(text('SELECT COUNT(*) FROM var_records')).scalar()

    print(f'  portfolio_snapshot: {snap_before:,} → {snap_after:,} (+{snap_after-snap_before:,})')
    print(f'  risk_logs:          {risk_before:,} → {risk_after:,} (+{risk_after-risk_before:,})')
    print(f'  var_records:        {var_before:,} → {var_after:,} (+{var_after-var_before:,})')

    # 安全验证
    fs_cnt = s.execute(text('SELECT COUNT(*) FROM fusion_score')).scalar()
    fv_cnt = s.execute(text('SELECT COUNT(*) FROM factor_value')).scalar()
    print(f'\n  fusion_score: {fs_cnt:,} (未被修改)')
    print(f'  factor_value: {fv_cnt:,} (未被修改)')

    # 绩效摘要
    if snapshots:
        final_val = float(snapshots[-1]['tv'])
        total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        sharpe = (mean(daily_rets) / stdev(daily_rets) * sqrt(252)) if len(daily_rets) > 1 and stdev(daily_rets) > 0 else 0
        print(f'\n  初始资金: {INITIAL_CAPITAL:,.0f}')
        print(f'  终值:     {final_val:,.0f}')
        print(f'  总收益:   {total_ret:+.2f}%')
        print(f'  夏普:     {sharpe:.3f}')
        print(f'  最大回撤: {max((ev["dd"] for ev in risk_events), default=0):.1f}%')
        print(f'  风控事件: {len(risk_events)}')

print('\n[OK] 简化回测完成')
