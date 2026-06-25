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
