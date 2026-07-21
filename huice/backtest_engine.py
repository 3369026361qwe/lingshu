"""事件驱动回测引擎 — 每日推进+调仓+记录+实验追踪。"""
import logging
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean as _stat_mean
from statistics import stdev as _stat_stdev

from shuju.utils import safe_divide


@dataclass
class PerformanceReport:
    """回测绩效报告 — 统一输出格式."""
    # 收益指标
    total_return: Decimal = Decimal("0")
    annualized_return: Decimal = Decimal("0")
    annualized_volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")

    # 风险指标
    var_95: Decimal = Decimal("0")
    var_99: Decimal = Decimal("0")
    cvar_95: Decimal = Decimal("0")
    cvar_99: Decimal = Decimal("0")

    # VaR 回测结果
    var_backtest: dict | None = None

    # 防数据窥探指标
    deflated_sharpe_ratio: float = 0.0
    probabilistic_sharpe_ratio: float = 0.0

    # 归因
    attribution: dict | None = None

_log = logging.getLogger(__name__)


class BacktestEngine:
    """事件驱动回测引擎 v4.0.

    集成 MockBroker + OrderManager 实现真实交易模拟:
        - 含佣金、滑点、印花税
        - T+1 限制
        - 按手取整 (A股)

    集成精算引擎:
        - VaRBacktestSuite: Kupiec/Christoffersen 检验
        - DataSnoopingDefender: DSR/PSR 防过拟合
        - AttributionEngine: 三维归因分析
    """

    def __init__(self, repository=None):
        self._repo = repository
        self._experiment_id = ""
        self._records: list[dict] = []
        self._trades: list[dict] = []
        self._config: dict = {}
        self._broker = None
        self._order_manager = None
        self._use_broker_sim = False

    # ── 运行回测 ────────────────────────────────────────

    def run(self, config: dict) -> dict:
        """执行回测.

        Args:
            config: {
                start_date, end_date, initial_capital, strategy_name, params,
                data_loader, signal_generator, executor,
                # v4.0 可选:
                use_broker_sim: bool (是否使用 MockBroker + OrderManager),
                commission_rate: Decimal,
                slippage: Decimal,
                enable_var_backtest: bool (是否运行 VaR 回测检验),
                enable_snooping_defense: bool (是否计算 DSR/PSR),
                enable_attribution: bool (是否运行归因分析),
                benchmark_weights: dict (归因分析的基准权重),
                benchmark_returns: dict (归因分析的基准收益),
            }

        Returns:
            完整实验报告 (含 VaR 回测, DSR/PSR, 归因分析)
        """
        self._experiment_id = (
            f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid.uuid4().hex[:8]}"
        )
        self._config = config
        t0 = _time.perf_counter()

        self._use_broker_sim = config.get("use_broker_sim", False)
        capital = Decimal(str(config.get("initial_capital", 1000000)))
        dates = config["data_loader"].get_trade_dates(config["start_date"], config["end_date"])

        # 初始化券商模拟
        if self._use_broker_sim:
            from zhixing.mock_broker import MockBroker
            from zhixing.order_manager import OrderManager

            self._order_manager = OrderManager()
            self._broker = MockBroker(
                commission_rate=config.get("commission_rate", Decimal("0.0003")),
                slippage=config.get("slippage", Decimal("0.0001")),
            )
            self._broker.cash = capital
            positions: dict[str, dict] = {}
        else:
            positions: dict[str, dict] = {}
            self._order_manager = None
            self._broker = None

        daily_records = []
        trades = []
        var_forecasts: list[Decimal] = []
        actual_losses: list[Decimal] = []
        prev_equity = capital
        daily_return_history: list[Decimal] = []

        # 归因追踪: 记录每期组合权重和收益
        attribution_snapshots: list[dict] = []
        portfolio_weight_history: dict[str, list[Decimal]] = {}
        benchmark_weight_history: dict[str, list[Decimal]] = {}
        portfolio_price_history: dict[str, list[Decimal]] = {}   # {code: [price_t]}
        benchmark_price_history: dict[str, list[Decimal]] = {}
        industry_map: dict[str, str] = config.get("industry_map", {})
        # 因子归因: 追踪每期组合因子暴露和收益
        factor_data: dict[str, list[Decimal]] = config.get("factor_data", {})  # {name: [f_t]}
        factor_exposure_snapshots: list[dict[str, Decimal]] = []  # [{name: exposure_t}]
        asset_factor_exposures: dict[str, dict[str, Decimal]] = config.get("asset_factor_exposures", {})  # {code: {factor_name: beta}}

        for _i, trade_date in enumerate(dates):
            # 获取当日行情
            market_data = config["data_loader"].load_market_data(trade_date)
            if not market_data:
                continue

            # 日终清算 (T+1)
            if self._broker and self._use_broker_sim:
                self._broker.end_of_day()

            # 生成信号
            if self._use_broker_sim:
                signals = config["signal_generator"].generate(
                    trade_date, market_data, self._broker.positions
                )
            else:
                signals = config["signal_generator"].generate(
                    trade_date, market_data, positions
                )

            # 执行调仓
            if signals:
                if self._use_broker_sim and self._order_manager and self._broker:
                    day_trades = self._execute_via_broker(signals, market_data)
                else:
                    day_trades = config["executor"].execute(
                        signals, positions, capital, market_data
                    )
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
                trades.extend(day_trades)

            # 计算当日市值
            market_value = Decimal("0")
            if self._use_broker_sim and self._broker:
                prices = {code: Decimal(str(bar.get("close", 0))) for code, bar in market_data.items()}
                total_equity = self._broker.total_equity(prices)
                cash_val = self._broker.cash
            else:
                for code, pos in positions.items():
                    bar = market_data.get(code, {})
                    close = Decimal(str(bar.get("close", 0)))
                    market_value += close * pos["quantity"]
                total_equity = capital + market_value
                cash_val = capital

            daily_record = {
                "date": trade_date,
                "cash": float(cash_val),
                "market_value": float(total_equity - cash_val),
                "total_equity": float(total_equity),
                "positions": len(self._broker.positions) if self._use_broker_sim and self._broker else len(positions),
            }
            daily_records.append(daily_record)

            # ── VaR 预测: 滚动窗口参数法 ──────────────
            if prev_equity > 0:
                daily_ret = (total_equity - prev_equity) / prev_equity
                daily_return_history.append(daily_ret)

                # 记录实际损失 (用于 VaR 回测)
                actual_loss = -daily_ret
                actual_losses.append(actual_loss)

                # 滚动窗口 VaR 预测: 用过去 N 个交易日收益的分布估计
                var_window = 60  # 使用 60 日滚动窗口
                if len(daily_return_history) >= var_window:
                    window_rets = daily_return_history[-var_window:]
                    mu_r = Decimal(str(_stat_mean([float(r) for r in window_rets])))
                    sigma_r = Decimal(str(_stat_stdev([float(r) for r in window_rets])))
                    if sigma_r == 0:
                        sigma_r = Decimal("0.01")
                    # 95% 参数 VaR: VaR = -(μ - 1.645·σ)
                    var_95_forecast = -(mu_r - Decimal("1.645") * sigma_r)
                else:
                    # 数据不足时用保守估计 (日 VaR ≈ 3%)
                    var_95_forecast = Decimal("0.03")

                var_forecasts.append(var_95_forecast)

            # ── 归因快照: 记录当日权重和收益 ──────────
            if config.get("enable_attribution", False):
                price_map = {}
                for c_code, bar in market_data.items():
                    close_val = bar.get("close")
                    if close_val is not None and float(close_val) > 0:
                        price_map[c_code] = Decimal(str(close_val))

                # 组合权重: 基于持仓市值
                total_mv = Decimal("0")
                pos_weights: dict[str, Decimal] = {}
                pos_set = self._broker.positions if self._use_broker_sim and self._broker else positions
                for c_code, pd in pos_set.items():
                    qty = pd.get("quantity", 0) if isinstance(pd, dict) else pd.quantity
                    if qty > 0:
                        mv = price_map.get(c_code, Decimal("0")) * Decimal(str(qty))
                        total_mv += mv
                        pos_weights[c_code] = mv

                if total_mv > 0:
                    pw_normalized = {c: v / total_mv for c, v in pos_weights.items()}
                else:
                    pw_normalized = {}

                # 基准权重: 等权或从 config 获取
                bw_normalized: dict[str, Decimal] = {}
                bm_weights = config.get("benchmark_weights", {})
                if isinstance(bm_weights, dict) and bm_weights:
                    bw_normalized = bm_weights.get(trade_date, {})
                if not bw_normalized and len(pos_weights) > 0:
                    bw_normalized = {c: Decimal("1") / Decimal(len(pos_weights)) for c in pos_weights}

                attribution_snapshots.append({
                    "date": trade_date,
                    "portfolio_weights": pw_normalized,
                    "benchmark_weights": bw_normalized,
                })

                for c_code, w in pw_normalized.items():
                    portfolio_weight_history.setdefault(c_code, []).append(w)
                    if c_code in price_map:
                        portfolio_price_history.setdefault(c_code, []).append(price_map[c_code])
                for c_code, w in bw_normalized.items():
                    benchmark_weight_history.setdefault(c_code, []).append(w)
                    if c_code in price_map:
                        benchmark_price_history.setdefault(c_code, []).append(price_map[c_code])

                # ── 因子暴露快照 ──────────────────────────
                if asset_factor_exposures and pw_normalized:
                    combined_exp: dict[str, Decimal] = {}
                    for c_code, w in pw_normalized.items():
                        for f_name, beta in asset_factor_exposures.get(c_code, {}).items():
                            combined_exp[f_name] = combined_exp.get(f_name, Decimal("0")) + w * beta
                    factor_exposure_snapshots.append(combined_exp)

            prev_equity = total_equity

        # ── 绩效计算 ────────────────────────────────────
        equity_curve = [Decimal(str(r["total_equity"])) for r in daily_records]
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_returns.append(
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                )

        from huice.performance_metrics import PerformanceMetrics
        metrics = PerformanceMetrics.compute_all(equity_curve, daily_returns)

        # ── VaR 回测检验 ────────────────────────────────
        var_backtest_result = None
        if config.get("enable_var_backtest", False) and len(var_forecasts) > 20:
            try:
                from jingsuan.var_backtest import VaRBacktestSuite
                # Align: actual_losses has n entries, var_forecasts starts at index 20
                n_actuals = len(actual_losses)
                n_forecasts = len(var_forecasts)
                if n_actuals >= 20 + n_forecasts:
                    aligned_losses = actual_losses[-n_forecasts:]
                else:
                    aligned_losses = actual_losses[20:20 + n_forecasts]
                if len(aligned_losses) == len(var_forecasts):
                    var_backtest_result = VaRBacktestSuite.run_all(
                        var_forecasts, aligned_losses, Decimal("0.95")
                    )
            except Exception as _exc:
                import logging
                logging.getLogger(__name__).warning("VaR backtest failed: %s", _exc)

        # ── 数据窥探防御 ──────────────────────────────────
        dsr, psr = 0.0, 0.0
        if config.get("enable_snooping_defense", False):
            try:
                from huice.data_snooping import DataSnoopingDefender
                sr = float(metrics.get("sharpe", 0))
                n_days = metrics.get("n_days", 252)
                n_trials = config.get("n_trials", 10)

                dsr = DataSnoopingDefender.deflated_sharpe_ratio(
                    sr, n_trials, max(n_days, 1)
                )
                psr = DataSnoopingDefender.probabilistic_sharpe_ratio(
                    sr, 0.0, max(n_days, 1)
                )
            except Exception as _exc:
                import logging
                logging.getLogger(__name__).warning("DSR/PSR computation failed: %s", _exc)

        enable_attribution = config.get("enable_attribution", False)
        # ── 归因分析 ────────────────────────────────────
        attribution_result = None
        if enable_attribution and len(attribution_snapshots) > 10:
            try:
                from huice.attribution import AttributionEngine

                attribution_result = {}

                # ── 1. Brinson 归因 ───────────────────────────
                last_snap = attribution_snapshots[-1]
                pw_by_sector: dict[str, list[tuple[str, Decimal]]] = {}
                bw_by_sector: dict[str, list[tuple[str, Decimal]]] = {}
                pr_map: dict[str, Decimal] = {}
                br_map: dict[str, Decimal] = {}

                for code, w in last_snap["portfolio_weights"].items():
                    sector = industry_map.get(code, "other")
                    pw_by_sector.setdefault(sector, []).append((code, w))
                    vals = portfolio_price_history.get(code, [])
                    pr_map[code] = (vals[-1] - vals[0]) / vals[0] if len(vals) > 1 and vals[0] > 0 else Decimal("0")
                for code, w in last_snap["benchmark_weights"].items():
                    sector = industry_map.get(code, "other")
                    bw_by_sector.setdefault(sector, []).append((code, w))
                    vals = benchmark_price_history.get(code, [])
                    br_map[code] = (vals[-1] - vals[0]) / vals[0] if len(vals) > 1 and vals[0] > 0 else Decimal("0")

                if pw_by_sector and bw_by_sector:
                    brinson_r = AttributionEngine.brinson(pw_by_sector, bw_by_sector, pr_map, br_map)
                    attribution_result["brinson"] = {
                        "sectors": brinson_r.sectors,
                        "allocation": [float(a) for a in brinson_r.allocation_effect],
                        "selection": [float(s) for s in brinson_r.selection_effect],
                        "interaction": [float(i) for i in brinson_r.interaction_effect],
                        "total_active": float(brinson_r.total_active_return),
                    }

                # ── 2. 因子归因 ──────────────────────────
                if factor_data and factor_exposure_snapshots and len(daily_returns) > 0:
                    n_factors = len(next(iter(factor_data.values()), []))
                    n_days = min(len(daily_returns), n_factors,
                                 len(factor_exposure_snapshots))
                    if n_days > 10:
                        # 聚合组合因子暴露: 每个因子取时间平均
                        agg_exposures: dict[str, list[Decimal]] = {}
                        for f_name in factor_data:
                            avg_exp = Decimal("0")
                            for snap in factor_exposure_snapshots:
                                avg_exp += snap.get(f_name, Decimal("0"))
                            avg_exp = safe_divide(avg_exp, Decimal(max(1, len(factor_exposure_snapshots))))
                            agg_exposures[f_name] = [avg_exp] * n_days

                        port_rets = [Decimal(str(r)) for r in daily_returns[-n_days:]]
                        factor_rets = {k: [Decimal(str(v)) for v in vs[-n_days:]] for k, vs in factor_data.items()}
                        try:
                            factor_r = AttributionEngine.factor_attribution(
                                port_rets, factor_rets, agg_exposures
                            )
                            attribution_result["factor"] = {
                                "contributions": {k: float(v) for k, v in factor_r.factor_contributions.items()},
                                "alpha": float(factor_r.alpha),
                                "r_squared": float(factor_r.r_squared),
                            }
                        except Exception as _exc:
                            import logging
                            logging.getLogger(__name__).warning("Factor attribution failed: %s", _exc)

                # ── 3. 风险归因 ──────────────────────────
                pw_list = [last_snap["portfolio_weights"].get(c, Decimal("0"))
                           for c in last_snap["portfolio_weights"]]
                if len(pw_list) > 1 and len(daily_returns) > 20:
                    from juece.portfolio_optimizer import PortfolioOptimizer
                    opt = PortfolioOptimizer()
                    n_a = len(pw_list)
                    # 用每个持仓的历史价格序列构建真实协方差矩阵
                    cov_rm = []
                    for code in last_snap["portfolio_weights"]:
                        if code in portfolio_price_history and len(portfolio_price_history[code]) >= 20:
                            vals = portfolio_price_history[code][-20:]
                            # 转为日收益率
                            rets = [(vals[i] - vals[i - 1]) / vals[i - 1] if i > 0 and vals[i - 1] > 0
                                    else Decimal("0") for i in range(1, len(vals))]
                            if len(rets) >= 2:
                                cov_rm.append(rets[:20])
                    # 回退: 若无法构建价格型协方差则用组合收益率 bootstrap
                    if len(cov_rm) < n_a:
                        cov_rm = [[Decimal(str(daily_returns[min(i, len(daily_returns) - 1)]))]
                                   * 20 for _ in range(n_a)]
                    else:
                        cov_rm = cov_rm[:n_a]
                    cov = opt.estimate_covariance(cov_rm)
                    risk_attr = AttributionEngine.risk_attribution(pw_list, cov)
                    attribution_result["risk"] = {
                        "var_total": float(risk_attr.var_total),
                        "component_var": [float(c) for c in risk_attr.component_var],
                        "marginal_var": [float(m) for m in risk_attr.marginal_var],
                    }
            except Exception as _exc:
                import logging
                logging.getLogger(__name__).warning("Attribution analysis failed: %s", _exc)

        elapsed = _time.perf_counter() - t0

        # ── 构建 PerformanceReport ──────────────────────
        report = PerformanceReport(
            total_return=Decimal(str(metrics.get("total_return", 0))),
            annualized_return=Decimal(str(metrics.get("annualized_return", 0))),
            annualized_volatility=Decimal(str(metrics.get("annualized_vol", 0))),
            sharpe_ratio=Decimal(str(metrics.get("sharpe", 0))),
            sortino_ratio=Decimal(str(metrics.get("sortino", 0))),
            calmar_ratio=Decimal(str(metrics.get("calmar", 0))),
            max_drawdown=Decimal(str(metrics.get("max_drawdown", 0))),
            win_rate=Decimal(str(metrics.get("win_rate", 0))),
            var_95=Decimal(str(metrics.get("var_95", 0))),
            var_99=Decimal(str(metrics.get("var_99", 0))),
            cvar_95=Decimal(str(metrics.get("cvar_95", 0))),
            cvar_99=Decimal(str(metrics.get("cvar_99", 0))),
            var_backtest=_var_result_to_dict(var_backtest_result) if var_backtest_result else None,
            deflated_sharpe_ratio=dsr,
            probabilistic_sharpe_ratio=psr,
            attribution=attribution_result,
        )

        full_report = {
            "experiment_id": self._experiment_id,
            "config": config,
            "metrics": metrics,
            "performance_report": _report_to_dict(report),
            "daily_records": daily_records,
            "trades": trades,
            "var_backtest": _var_result_to_dict(var_backtest_result),
            "dsr_pvalue": dsr,
            "psr": psr,
            "attribution": attribution_result,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._records = daily_records
        self._trades = trades

        # 持久化
        if self._repo:
            self._persist(full_report)

        return full_report

    # ── 券商模拟执行 ───────────────────────────────────

    def _execute_via_broker(self, signals: list[dict], market_data: dict) -> list[dict]:
        """通过 MockBroker + OrderManager 执行调仓."""
        trades = []
        for sig in signals:
            code = sig["code"]
            bar = market_data.get(code, {})
            price = Decimal(str(bar.get("close", 0)))
            if price <= 0:
                continue

            direction = sig.get("direction", "BUY")
            weight = Decimal(str(sig.get("weight", 0)))
            total_equity = self._broker.total_equity(
                {c: Decimal(str(market_data.get(c, {}).get("close", 0)))
                 for c in market_data}
            )

            if weight > 0 and direction == "BUY":
                amount = weight * total_equity
                qty = int(amount / price)
            elif direction == "SELL":
                pos = self._broker.positions.get(code, {})
                qty = int(pos.get("quantity", 0) * weight) if weight > 0 else pos.get("quantity", 0)
            else:
                qty = 0

            if qty < 100:
                continue

            try:
                order = self._order_manager.create(code, direction, qty, price, reason=sig.get("reason", ""))
                trade = self._broker.submit(order)
                if trade.get("status") != "rejected":
                    trades.append(trade)
            except Exception:
                continue

        return trades

    # ── 持久化 ─────────────────────────────────────────

    def _persist(self, report: dict) -> None:
        """持久化实验报告到数据库."""
        try:
            from sqlalchemy import text

            from shujuku.session import SessionContext

            with SessionContext() as s:
                snap_before = s.execute(text(
                    'SELECT COUNT(*) FROM portfolio_snapshot'
                )).scalar()

                batch = []
                for _rec_idx, rec in enumerate(self._records):
                    # 从 previous total_equity 计算 daily_return
                    prev_tv = Decimal(str(self._records[_rec_idx - 1]["total_equity"])) if _rec_idx > 0 else None
                    cur_tv = Decimal(str(rec["total_equity"]))
                    dr = (cur_tv - prev_tv) / prev_tv if prev_tv and prev_tv > 0 else Decimal("0")
                    cr = (cur_tv - Decimal(str(self._records[0]["total_equity"]))) / Decimal(str(self._records[0]["total_equity"])) \
                        if self._records and Decimal(str(self._records[0]["total_equity"])) > 0 else Decimal("0")

                    batch.append({
                        'd': datetime.strptime(rec["date"], "%Y%m%d").date(),
                        'tv': str(cur_tv),
                        'c': str(rec.get("cash", 0)),
                        'mv': str(rec.get("market_value", 0)),
                        'dr': str(dr),
                        'cr': str(cr),
                        'pc': rec.get("positions", 0),
                    })

                if batch:
                    stmt = text(
                        "INSERT INTO portfolio_snapshot (trade_date, total_value, cash, market_value, "
                        "daily_return, cumulative_return, position_count, updated_at) "
                        "VALUES (:d, :tv, :c, :mv, :dr, :cr, :pc, datetime('now')) "
                        "ON CONFLICT(trade_date) DO NOTHING"
                    )
                    for i in range(0, len(batch), 2000):
                        s.execute(stmt, batch[i:i + 2000])
                    s.commit()

                    snap_after = s.execute(text(
                        'SELECT COUNT(*) FROM portfolio_snapshot'
                    )).scalar()
                    import logging
                    logging.getLogger(__name__).info(
                        "portfolio_snapshot persisted: %s -> %s (+%s rows)",
                        snap_before, snap_after, snap_after - snap_before,
                    )
        except Exception as exc:
            _log.warning("Persist failed: %s", exc)

    # ── 实验对比 ────────────────────────────────────────

    @staticmethod
    def compare_experiments(reports: list[dict]) -> dict:
        """对比多个实验的绩效指标."""
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
                "dsr_pvalue": r.get("dsr_pvalue"),
                "psr": r.get("psr"),
                "var_backtest": r.get("var_backtest"),
            })
            if m.get("sharpe", -999) and float(m["sharpe"]) > best_sharpe:
                best_sharpe = float(m["sharpe"])
                comparison["best_sharpe"] = r["experiment_id"]
            if m.get("total_return", -999) and float(m["total_return"]) > best_return:
                best_return = float(m["total_return"])
                comparison["best_return"] = r["experiment_id"]
        return comparison

    @property
    def experiment_id(self) -> str:
        return self._experiment_id


# ── DB-Backed Runner ──────────────────────────────────────────

class DBBacktestRunner:
    """从数据库加载行情和信号，运行回测并持久化结果的便捷运行器 (v4.0).

    v4.0 新增:
        - VaR 回测 (Kupiec/Christoffersen)
        - DSR/PSR 数据窥探防御
        - 归因分析输出
    """

    def __init__(self, repository=None):
        self._repo = repository
        self._engine = BacktestEngine(repository)

    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        top_n: int = 20,
        rebalance_freq: int = 40,
        initial_capital: float = 1_000_000,
        signal_source: str = 'fusion_score',
        risk_config: dict = None,
        persist: bool = True,
        enable_var_backtest: bool = False,
        enable_snooping_defense: bool = False,
        n_trials: int = 10,
    ) -> dict:
        """执行一次完整的 DB-backed 回测.

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期
            top_n: 持仓股票数量
            rebalance_freq: 调仓间隔（交易日）
            initial_capital: 初始资金
            signal_source: 信号来源
            risk_config: 风控配置
            persist: 是否持久化到 DB
            enable_var_backtest: 启用 VaR 回测检验
            enable_snooping_defense: 启用 DSR/PSR
            n_trials: DSR 的试验次数

        Returns:
            完整报告含 VaR 回测, DSR/PSR
        """
        import time as _time
        from collections import defaultdict
        from math import sqrt as _math_sqrt
        from statistics import mean, stdev

        from sqlalchemy import text

        from shujuku.session import SessionContext

        t0 = _time.perf_counter()

        with SessionContext() as s:
            prices = defaultdict(dict)
            date_filter = ""
            params = {}
            if start_date and end_date:
                date_filter = " WHERE trade_date BETWEEN :start AND :end"
                params = {"start": start_date, "end": end_date}
            elif start_date:
                date_filter = " WHERE trade_date >= :start"
                params = {"start": start_date}
            elif end_date:
                date_filter = " WHERE trade_date <= :end"
                params = {"end": end_date}

            pr_rows = s.execute(text(
                f"SELECT code, trade_date, close FROM daily_bar{date_filter} ORDER BY code, trade_date"
            ), params).fetchall()
            for code, td, close in pr_rows:
                try:
                    prices[str(td)][code] = float(str(close))
                except Exception:
                    pass

            all_dates = sorted(prices.keys())

            signals_data = defaultdict(dict)
            table = 'fusion_score' if signal_source == 'fusion_score' else 'factor_value'
            if signal_source == 'fusion_score':
                fs_rows = s.execute(text(
                    f"SELECT trade_date, code, composite_score, rank FROM {table}{date_filter} ORDER BY trade_date, rank"
                ), params).fetchall()
                for td, code, sc, _rk in fs_rows:
                    try:
                        signals_data[str(td)][code] = float(sc)
                    except Exception:
                        pass
            else:
                fs_rows = s.execute(text(
                    f"SELECT trade_date, code, z_score FROM {table}{date_filter} ORDER BY trade_date, z_score DESC"
                ), params).fetchall()
                for td, code, sc in fs_rows:
                    try:
                        signals_data[str(td)][code] = float(sc)
                    except Exception:
                        pass

        # 回测循环
        cash = initial_capital
        holdings = {}
        snapshots = []
        daily_rets = []
        risk_events = []
        var_values = []
        var_forecasts = []
        actual_losses = []
        peak_value = initial_capital
        rebal_day = 0

        for trade_date in all_dates:
            day_prices = prices.get(trade_date, {})
            if not day_prices:
                continue

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

            market_value = cash
            for code, qty in holdings.items():
                px = day_prices.get(code)
                if px:
                    market_value += qty * px

            total_value = market_value
            prev_val = float(snapshots[-1]['tv']) if snapshots else initial_capital
            daily_ret = (total_value - prev_val) / prev_val if prev_val > 0 else 0
            daily_rets.append(daily_ret)
            actual_losses.append(-daily_ret)

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

        # VaR 计算 — 与 actual_losses 同步追加 var_forecasts
        if len(daily_rets) >= 20:
            for i in range(20, len(daily_rets)):
                window = daily_rets[i - 20:i]
                mu_val = mean(window)
                sigma_val = stdev(window) if len(window) > 1 else 0.01
                var_95_pct = mu_val - 1.645 * sigma_val
                cvar_vals = [r for r in window if r <= var_95_pct]
                cvar_95_pct = mean(cvar_vals) if cvar_vals else var_95_pct
                curr_val = float(snapshots[i]['tv']) if i < len(snapshots) else initial_capital
                var_values.append({
                    'd': all_dates[i] + 'T15:00:00',
                    'var95': str(round(var_95_pct * curr_val, 2)),
                    'cvar95': str(round(cvar_95_pct * curr_val, 2)),
                })
                var_forecasts.append(Decimal(str(var_95_pct)))
            # 对齐: actual_losses 和 var_forecasts 都从第 20 天开始
            actual_losses_aligned = actual_losses[20:]

        # 绩效指标
        sharpe = 0.0
        if len(daily_rets) > 1:
            sigma_d = stdev(daily_rets)
            if sigma_d > 0:
                sharpe = mean(daily_rets) / sigma_d * _math_sqrt(252)

        final_val = float(snapshots[-1]['tv']) if snapshots else initial_capital
        total_ret = (final_val - initial_capital) / initial_capital * 100
        max_dd = max((ev['dd'] for ev in risk_events), default=0)

        # ── VaR 回测检验 ────────────────────────────────
        var_bt_result = None
        if enable_var_backtest and len(var_forecasts) > 20 and len(actual_losses_aligned) > 20:
            try:
                from jingsuan.var_backtest import VaRBacktestSuite
                result = VaRBacktestSuite.run_all(
                    var_forecasts,
                    [Decimal(str(l)) for l in actual_losses_aligned],
                    Decimal("0.95"),
                )
                var_bt_result = {
                    "n_observations": result.n_observations,
                    "n_violations": result.n_violations,
                    "violation_rate": float(result.violation_rate),
                    "kupiec_pvalue": result.kupiec_pvalue,
                    "kupiec_pass": result.kupiec_pass,
                    "christoffersen_pvalue": result.christoffersen_pvalue,
                    "christoffersen_pass": result.christoffersen_pass,
                    "basel_zone": result.basel_zone,
                    "basel_multiplier": float(result.basel_multiplier),
                }
            except Exception:
                pass

        # ── 数据窥探防御 ──────────────────────────────────
        dsr_pvalue = 0.0
        psr = 0.0
        if enable_snooping_defense and len(daily_rets) > 1:
            try:
                from huice.data_snooping import DataSnoopingDefender
                dsr_pvalue = DataSnoopingDefender.deflated_sharpe_ratio(
                    sharpe, n_trials, len(daily_rets)
                )
                psr = DataSnoopingDefender.probabilistic_sharpe_ratio(
                    sharpe, 0.0, len(daily_rets)
                )
            except Exception:
                pass

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
            'var_backtest': var_bt_result,
            'dsr_pvalue': dsr_pvalue,
            'psr': psr,
        }

        if persist:
            self._persist_results(snapshots, risk_events, var_values)

        return report

    def _persist_results(self, snapshots: list, risk_events: list, var_values: list) -> None:
        """持久化回测结果到 DB."""
        try:
            from sqlalchemy import text

            from shujuku.session import SessionContext

            with SessionContext() as s:
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
            _log.warning("Persist failed: %s", exc)

    @staticmethod
    def print_summary(report: dict) -> None:
        """打印回测绩效摘要."""
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
        if report.get('dsr_pvalue'):
            print(f'  DSR p值:  {report["dsr_pvalue"]:.4f}')
        if report.get('psr'):
            print(f'  PSR:      {report["psr"]:.4f}')
        if report.get('var_backtest'):
            vb = report['var_backtest']
            print(f'  VaR检验:  Kupiec p={vb["kupiec_pvalue"]:.3f}, Basel={vb["basel_zone"]}')
        print('=' * 60)


# ── 辅助函数 ──────────────────────────────────────────

def _var_result_to_dict(result) -> dict | None:
    """VaRBacktestResult → dict."""
    if result is None:
        return None
    return {
        "n_observations": result.n_observations,
        "n_violations": result.n_violations,
        "violation_rate": float(result.violation_rate),
        "expected_violations": float(result.expected_violations),
        "kupiec_lr": result.kupiec_lr,
        "kupiec_pvalue": result.kupiec_pvalue,
        "kupiec_pass": result.kupiec_pass,
        "christoffersen_ind_lr": result.christoffersen_ind_lr,
        "christoffersen_cc_lr": result.christoffersen_cc_lr,
        "christoffersen_pvalue": result.christoffersen_pvalue,
        "christoffersen_pass": result.christoffersen_pass,
        "basel_zone": result.basel_zone,
        "basel_multiplier": float(result.basel_multiplier),
    }


def _report_to_dict(report: PerformanceReport) -> dict:
    """PerformanceReport → dict."""
    return {
        "total_return": float(report.total_return),
        "annualized_return": float(report.annualized_return),
        "annualized_volatility": float(report.annualized_volatility),
        "sharpe_ratio": float(report.sharpe_ratio),
        "sortino_ratio": float(report.sortino_ratio),
        "calmar_ratio": float(report.calmar_ratio),
        "max_drawdown": float(report.max_drawdown),
        "win_rate": float(report.win_rate),
        "var_95": float(report.var_95),
        "var_99": float(report.var_99),
        "cvar_95": float(report.cvar_95),
        "cvar_99": float(report.cvar_99),
        "deflated_sharpe_ratio": report.deflated_sharpe_ratio,
        "probabilistic_sharpe_ratio": report.probabilistic_sharpe_ratio,
    }
