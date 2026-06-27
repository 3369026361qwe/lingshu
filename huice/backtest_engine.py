"""事件驱动回测引擎 — 每日推进+调仓+记录+实验追踪。"""
import json
import uuid
import time as _time
from datetime import datetime, timezone, date
from decimal import Decimal
from typing import Optional

from huice.performance_metrics import PerformanceMetrics


class BacktestEngine:
    """事件驱动回测引擎。每个实验自动生成唯一ID并记录全部数据。"""

    def __init__(self, repository=None):
        self._repo = repository
        self._experiment_id = ""
        self._records: list[dict] = []  # 每日记录
        self._trades: list[dict] = []   # 成交记录
        self._config: dict = {}

    # ── 运行回测 ────────────────────────────────────────

    def run(self, config: dict) -> dict:
        """执行回测。

        Args:
            config: {start_date, end_date, initial_capital, strategy_name, params,
                     data_loader, signal_generator, executor}

        Returns:
            完整实验报告 (可持久化)
        """
        self._experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._config = config
        t0 = _time.perf_counter()

        capital = Decimal(str(config.get("initial_capital", 1000000)))
        dates = config["data_loader"].get_trade_dates(config["start_date"], config["end_date"])
        positions: dict[str, dict] = {}  # {code: {quantity, avg_cost}}
        daily_records = []
        trades = []

        for i, trade_date in enumerate(dates):
            # 获取当日行情
            market_data = config["data_loader"].load_market_data(trade_date)
            if not market_data:
                continue

            # 生成信号
            signals = config["signal_generator"].generate(trade_date, market_data, positions)

            # 执行调仓
            if signals:
                day_trades = config["executor"].execute(signals, positions, capital, market_data)
                trades.extend(day_trades)
                for t in day_trades:
                    if t["direction"] == "BUY":
                        capital -= Decimal(str(t["amount"]))
                        pos = positions.get(t["code"], {"quantity": 0, "avg_cost": Decimal("0")})
                        total_cost = pos["avg_cost"] * pos["quantity"] + Decimal(str(t["amount"]))
                        pos["quantity"] += t["quantity"]
                        pos["avg_cost"] = total_cost / pos["quantity"] if pos["quantity"] > 0 else Decimal("0")
                        positions[t["code"]] = pos
                    else:
                        capital += Decimal(str(t["amount"]))
                        pos = positions.get(t["code"])
                        if pos:
                            pos["quantity"] -= t["quantity"]
                            if pos["quantity"] <= 0:
                                positions.pop(t["code"], None)

            # 计算当日市值
            market_value = Decimal("0")
            for code, pos in positions.items():
                bar = market_data.get(code, {})
                close = Decimal(str(bar.get("close", 0)))
                market_value += close * pos["quantity"]

            total_equity = capital + market_value
            daily_records.append({
                "date": trade_date, "cash": float(capital), "market_value": float(market_value),
                "total_equity": float(total_equity), "positions": len(positions),
            })

        # 绩效计算
        equity_curve = [Decimal(str(r["total_equity"])) for r in daily_records]
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])

        metrics = PerformanceMetrics.compute_all(equity_curve, daily_returns)

        elapsed = _time.perf_counter() - t0
        report = {
            "experiment_id": self._experiment_id,
            "config": config,
            "metrics": metrics,
            "daily_records": daily_records,
            "trades": trades,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._records = daily_records
        self._trades = trades

        # 持久化
        if self._repo:
            self._persist(report)

        return report

    # ── 持久化实验数据 ───────────────────────────────────

    def _persist(self, report: dict) -> None:
        """持久化实验报告到数据库。ON CONFLICT DO NOTHING 保护已有快照。"""
        try:
            from shujuku.session import SessionContext
            from sqlalchemy import text

            with SessionContext() as s:
                for rec in self._records:
                    s.execute(text(
                        "INSERT INTO portfolio_snapshot (trade_date, total_value, cash, market_value, "
                        "daily_return, cumulative_return, position_count, updated_at) "
                        "VALUES (:d, :tv, :c, :mv, :dr, :cr, :pc, datetime('now')) "
                        "ON CONFLICT(trade_date) DO NOTHING"
                    ), {
                        'd': datetime.strptime(rec["date"], "%Y%m%d").date(),
                        'tv': str(rec["total_equity"]),
                        'c': str(rec.get("cash", 0)),
                        'mv': str(rec.get("market_value", 0)),
                        'dr': str(rec.get("daily_return", 0) or 0),
                        'cr': str(rec.get("cumulative_return", 0) or 0),
                        'pc': rec.get("positions", 0),
                    })
                s.commit()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Persist failed: %s", exc)

    # ── 实验对比 ────────────────────────────────────────

    @staticmethod
    def compare_experiments(reports: list[dict]) -> dict:
        """对比多个实验的绩效指标。"""
        if not reports:
            return {}
        comparison = {"experiments": [], "best_sharpe": "", "best_return": ""}
        best_sharpe = -999.0
        best_return = -999.0
        for r in reports:
            m = r.get("metrics", {})
            comparison["experiments"].append({
                "experiment_id": r.get("experiment_id", ""),
                "sharpe": m.get("sharpe"),
                "total_return": m.get("total_return"),
                "max_drawdown": m.get("max_drawdown"),
                "win_rate": m.get("win_rate"),
            })
            if m.get("sharpe", -999) and float(m["sharpe"]) > best_sharpe:
                best_sharpe = float(m["sharpe"]); comparison["best_sharpe"] = r["experiment_id"]
            if m.get("total_return", -999) and float(m["total_return"]) > best_return:
                best_return = float(m["total_return"]); comparison["best_return"] = r["experiment_id"]
        return comparison

    @property
    def experiment_id(self) -> str:
        return self._experiment_id


# ── DB-Backed Runner ────────────────────────────────────────────────────────

class DBBacktestRunner:
    """从数据库加载行情和信号，运行回测并持久化结果的便捷运行器。

    封装了所有回测脚本共用的 DB 查询、回测循环、风控检测、VaR 计算、
    结果持久化和性能摘要逻辑。脚本只需提供参数即可。

    Usage:
        runner = DBBacktestRunner()
        report = runner.run(
            start_date='20240101', end_date='20241231',
            top_n=20, rebalance_freq=40,
            initial_capital=1_000_000,
        )
    """

    def __init__(self, repository=None):
        self._repo = repository
        self._engine = BacktestEngine(repository)

    # ── 主入口 ──────────────────────────────────────────

    def run(self, start_date: str = None, end_date: str = None,
            top_n: int = 20, rebalance_freq: int = 40,
            initial_capital: float = 1_000_000,
            signal_source: str = 'fusion_score',
            risk_config: dict = None,
            persist: bool = True,
            ) -> dict:
        """执行一次完整的 DB-backed 回测。

        Args:
            start_date: 开始日期 (YYYYMMDD), None=全部
            end_date: 结束日期, None=全部
            top_n: 持仓股票数量
            rebalance_freq: 调仓间隔（交易日）
            initial_capital: 初始资金
            signal_source: 信号来源 ('fusion_score' | 'factor_value')
            risk_config: 风控配置
            persist: 是否持久化到 DB

        Returns:
            包含 metrics, daily_records, trades, risk_events, var_records 的报告
        """
        from shujuku.session import SessionContext
        from sqlalchemy import text
        from collections import defaultdict
        import time as _time
        from math import sqrt
        from statistics import mean, stdev

        t0 = _time.perf_counter()

        with SessionContext() as s:
            # 1. 加载价格数据
            prices = defaultdict(dict)
            pr_rows = s.execute(text(
                'SELECT code, trade_date, close FROM daily_bar ORDER BY code, trade_date'
            )).fetchall()
            for code, td, close in pr_rows:
                try:
                    prices[str(td)][code] = float(str(close))
                except Exception:
                    pass

            all_dates = sorted(prices.keys())
            if start_date:
                all_dates = [d for d in all_dates if d >= start_date]
            if end_date:
                all_dates = [d for d in all_dates if d <= end_date]

            # 2. 加载信号
            signals_data = defaultdict(dict)
            table = 'fusion_score' if signal_source == 'fusion_score' else 'factor_value'
            if signal_source == 'fusion_score':
                fs_rows = s.execute(text(
                    f'SELECT trade_date, code, composite_score, rank FROM {table} ORDER BY trade_date, rank'
                )).fetchall()
                for td, code, sc, rk in fs_rows:
                    try:
                        signals_data[str(td)][code] = float(sc)
                    except Exception:
                        pass
            else:
                # factor_value 没有 rank 列 — 按 z_score DESC + ROW_NUMBER 计算
                fs_rows = s.execute(text(
                    f'SELECT trade_date, code, z_score FROM {table} ORDER BY trade_date, z_score DESC'
                )).fetchall()
                for td, code, sc in fs_rows:
                    try:
                        signals_data[str(td)][code] = float(sc)
                    except Exception:
                        pass

        # 3. 运行回测循环
        cash = initial_capital
        holdings = {}
        snapshots = []
        daily_rets = []
        risk_events = []
        var_values = []
        peak_value = initial_capital
        rebal_day = 0

        for trade_date in all_dates:
            day_prices = prices.get(trade_date, {})
            if not day_prices:
                continue

            # 调仓
            rebal_day += 1
            if rebal_day >= rebalance_freq or not holdings:
                rebal_day = 0
                # 卖出全部
                for code, qty in list(holdings.items()):
                    px = day_prices.get(code)
                    if px and qty > 0:
                        cash += qty * px
                    del holdings[code]

                # 买入 Top N
                picks = signals_data.get(trade_date, {})
                ranked = sorted(picks.items(), key=lambda x: x[1], reverse=True)[:top_n]
                if ranked:
                    weight_per_stock = cash / top_n
                    for code, _score in ranked:
                        px = day_prices.get(code)
                        if px and px > 0:
                            qty = int(weight_per_stock / px / 100) * 100
                            if qty > 0:
                                holdings[code] = qty
                                cash -= qty * px

            # 市值
            market_value = cash
            for code, qty in holdings.items():
                px = day_prices.get(code)
                if px:
                    market_value += qty * px

            total_value = market_value

            # 日收益率
            prev_val = float(snapshots[-1]['tv']) if snapshots else initial_capital
            daily_ret = (total_value - prev_val) / prev_val if prev_val > 0 else 0
            daily_rets.append(daily_ret)

            # 回撤检测
            if total_value > peak_value:
                peak_value = total_value
            drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0

            level = 'LOW'
            if drawdown > 0.20:
                level = 'CRITICAL'
            elif drawdown > 0.15:
                level = 'HIGH'
            elif drawdown > 0.10:
                level = 'MEDIUM'

            if level in ('CRITICAL', 'HIGH'):
                risk_events.append({
                    'd': trade_date, 'dd': round(drawdown * 100, 1), 'lvl': level,
                    'scale': round(1.0 - drawdown, 2),
                })

            cum_ret = (total_value - initial_capital) / initial_capital
            snapshots.append({
                'd': trade_date, 'tv': str(round(total_value, 2)),
                'cash': str(round(cash, 2)), 'mv': str(round(total_value - cash, 2)),
                'dr': str(round(daily_ret, 8)), 'cr': str(round(cum_ret, 8)),
                'pc': len(holdings),
            })

        # 4. VaR 计算
        if len(daily_rets) >= 20:
            for i in range(20, len(daily_rets)):
                window = daily_rets[i - 20:i]
                mu = mean(window)
                sigma = stdev(window) if len(window) > 1 else 0.01
                var_95_pct = mu - 1.645 * sigma
                cvar_vals = [r for r in window if r <= var_95_pct]
                cvar_95_pct = mean(cvar_vals) if cvar_vals else var_95_pct
                curr_val = float(snapshots[i]['tv']) if i < len(snapshots) else initial_capital
                var_values.append({
                    'd': all_dates[i] + 'T15:00:00',
                    'var95': str(round(var_95_pct * curr_val, 2)),
                    'cvar95': str(round(cvar_95_pct * curr_val, 2)),
                })

        # 5. 绩效指标
        sharpe = 0.0
        if len(daily_rets) > 1:
            sigma_d = stdev(daily_rets)
            if sigma_d > 0:
                sharpe = mean(daily_rets) / sigma_d * sqrt(252)

        final_val = float(snapshots[-1]['tv']) if snapshots else initial_capital
        total_ret = (final_val - initial_capital) / initial_capital * 100
        max_dd = max((ev['dd'] for ev in risk_events), default=0)

        elapsed = _time.perf_counter() - t0

        report = {
            'start_date': all_dates[0] if all_dates else '',
            'end_date': all_dates[-1] if all_dates else '',
            'trading_days': len(all_dates),
            'initial_capital': initial_capital,
            'final_value': final_val,
            'total_return_pct': round(total_ret, 2),
            'sharpe': round(sharpe, 3),
            'max_drawdown_pct': max_dd,
            'risk_events': len(risk_events),
            'snapshots': snapshots,
            'risk_events_list': risk_events,
            'var_records': var_values,
            'daily_returns': daily_rets,
            'elapsed_seconds': round(elapsed, 1),
        }

        # 6. 持久化
        if persist:
            self._persist_results(snapshots, risk_events, var_values)

        return report

    # ── 持久化 ──────────────────────────────────────────

    def _persist_results(self, snapshots: list, risk_events: list, var_values: list) -> None:
        """持久化回测结果到 DB。INSERT ON CONFLICT DO NOTHING 保护已有数据。"""
        try:
            from shujuku.session import SessionContext
            from sqlalchemy import text

            with SessionContext() as s:
                snap_before = s.execute(text('SELECT COUNT(*) FROM portfolio_snapshot')).scalar()

                if snapshots:
                    stmt = text(
                        'INSERT INTO portfolio_snapshot (trade_date, total_value, cash, market_value, '
                        'daily_return, cumulative_return, position_count, updated_at) '
                        'VALUES (:d, :tv, :cash, :mv, :dr, :cr, :pc, datetime("now")) '
                        'ON CONFLICT(trade_date) DO NOTHING'
                    )
                    for i in range(0, len(snapshots), 2000):
                        s.execute(stmt, snapshots[i:i + 2000])
                    s.commit()

                snap_after = s.execute(text('SELECT COUNT(*) FROM portfolio_snapshot')).scalar()

                import logging
                _log = logging.getLogger(__name__)
                _log.info('portfolio_snapshot: %s → %s (+%s)',
                          snap_before, snap_after, snap_after - snap_before)

                # Risk logs
                if risk_events:
                    existing = set(r[0] for r in s.execute(text(
                        'SELECT DISTINCT timestamp FROM risk_logs'
                    )).fetchall())
                    new_events = [ev for ev in risk_events
                                  if (ev['d'] + 'T15:00:00') not in existing]
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
                                    'd': f"date={ev['d']} drawdown={ev['dd']}%"
                                })
                            except Exception:
                                pass
                    s.commit()

                # VaR records
                if var_values:
                    existing_dates = set(r[0] for r in s.execute(text(
                        'SELECT DISTINCT calc_date FROM var_records'
                    )).fetchall())
                    new_vars = [v for v in var_values if v['d'] not in existing_dates]
                    if new_vars:
                        stmt = text(
                            "INSERT INTO var_records (calc_date, confidence_level, var, cvar, "
                            "method, window_days, created_at) "
                            "VALUES (:d, 0.95, :var95, :cvar95, 'historical', 20, datetime('now'))"
                        )
                        for i in range(0, len(new_vars), 2000):
                            s.execute(stmt, new_vars[i:i + 2000])
                        s.commit()

        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Persist failed: %s", exc)

    # ── 摘要打印 ────────────────────────────────────────

    @staticmethod
    def print_summary(report: dict) -> None:
        """打印回测绩效摘要。"""
        print()
        print('=' * 60)
        print(f'回测完成 | {report["start_date"]} ~ {report["end_date"]} | {report["trading_days"]} 天')
        print('=' * 60)
        print(f'  初始资金: {report["initial_capital"]:,.0f}')
        print(f'  终值:     {report["final_value"]:,.0f}')
        print(f'  总收益:   {report["total_return_pct"]:+.2f}%')
        print(f'  夏普:     {report["sharpe"]:.3f}')
        print(f'  最大回撤: {report["max_drawdown_pct"]:.1f}%')
        print(f'  风控事件: {report["risk_events"]}')
        print(f'  耗时:     {report["elapsed_seconds"]:.0f}s')
        print('=' * 60)
