"""回测路由 — 回测运行 + 绩效摘要查询。"""

from math import sqrt
from statistics import mean, stdev

from fastapi import APIRouter
from sqlalchemy import text

from huice.backtest_engine import BacktestEngine
from jiekou.dependencies import get_repository
from jiekou.schemas import BacktestRequest

router = APIRouter(prefix="/api", tags=["backtest"])


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """运行一次回测实验。"""
    engine = BacktestEngine()

    class SimpleLoader:
        def get_trade_dates(self, s, e):
            return [f"2026{(i // 30) + 1:02d}{(i % 30) + 1:02d}" for i in range(60)]

        def load_market_data(self, d):
            return {}

    class SimpleSignal:
        def generate(self, d, m, p):
            return None

    class SimpleExec:
        def execute(self, s, p, c, m):
            return []

    config = {
        "start_date": req.start_date,
        "end_date": req.end_date,
        "initial_capital": float(req.initial_capital),
        "strategy_name": "api_backtest",
        "params": req.strategy_params,
        "data_loader": SimpleLoader(),
        "signal_generator": SimpleSignal(),
        "executor": SimpleExec(),
    }
    import asyncio
    report = await asyncio.to_thread(engine.run, config)
    return {
        "experiment_id": report["experiment_id"],
        "metrics": report["metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


@router.get("/backtest/summary")
async def get_backtest_summary():
    """回测绩效摘要 — 从 portfolio_snapshot 表读取。"""
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(
            text(
                "SELECT trade_date, total_value, cumulative_return, daily_return, position_count "
                "FROM portfolio_snapshot ORDER BY trade_date"
            )
        ).fetchall()
        if not rows:
            return {"error": "no backtest data"}

        values = [r[1] for r in rows if r[1]]
        rets = [r[3] for r in rows if r[3] is not None]
        peak = max(values) if values else 0
        final = values[-1] if values else 0
        dd = min((v - peak) / peak for v in values) if peak > 0 else 0

        return {
            "start_date": str(rows[0][0]),
            "end_date": str(rows[-1][0]),
            "total_days": len(rows),
            "final_value": final,
            "total_return": (final - 1_000_000) / 1_000_000 if final else 0,
            "max_drawdown": dd,
            "sharpe": (mean(rets) / stdev(rets) * sqrt(252)) if len(rets) > 1 else 0,
            "snapshot_count": len(rows),
            "equity_curve": [
                {"date": str(r[0]), "value": r[1]} for r in rows if r[1] is not None
            ],
        }
