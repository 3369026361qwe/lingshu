"""FastAPI 入口 + REST端点 + WebSocket。"""
import json
from datetime import datetime, timezone, date
from decimal import Decimal
from math import sqrt
from statistics import mean, stdev

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from jiekou.schemas import SelectionRequest, BacktestRequest, ErrorResponse
from jiekou.dependencies import get_repository

app = FastAPI(title="灵枢 LingShu API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# WebSocket 连接管理
_ws_clients: list[WebSocket] = []


# ── REST 端点 ────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/stocks")
async def list_stocks():
    """全A股列表。"""
    repo = get_repository()
    stocks = repo.get_active_stocks()
    return [{"code": s.code, "name": s.name, "exchange": s.exchange} for s in stocks]


@app.get("/api/stocks/{code}")
async def get_stock(code: str):
    repo = get_repository()
    s = repo.get_stock_by_code(code)
    if not s: return JSONResponse({"error": "not found"}, 404)
    return {"code": s.code, "name": s.name, "exchange": s.exchange}


@app.get("/api/stocks/{code}/daily")
async def get_daily_bars(code: str, start: str = "20260101", end: str = "20260630"):
    repo = get_repository()
    bars = repo.get_daily_bars(code, date.fromisoformat(start[:4]+"-"+start[4:6]+"-"+start[6:]),
                               date.fromisoformat(end[:4]+"-"+end[4:6]+"-"+end[6:]))
    return [{"trade_date": str(b.trade_date), "open": str(b.open), "high": str(b.high),
             "low": str(b.low), "close": str(b.close), "volume": str(b.volume)} for b in bars]


@app.get("/api/selection")
async def stock_selection(date: str = "", top_n: int = 30):
    """选股接口 — 从 fusion_score 表读取最新选股。"""
    from sqlalchemy import text
    repo = get_repository()
    with repo._session as s:
        if not date:
            latest = s.execute(text('SELECT MAX(trade_date) FROM fusion_score')).scalar()
            date = str(latest) if latest else ''
        rows = s.execute(text(
            "SELECT code, composite_score, rank FROM fusion_score WHERE trade_date = :d ORDER BY rank LIMIT :n"
        ), {'d': date, 'n': top_n}).fetchall()
        picks = [{"code": r[0], "score": float(r[1]), "rank": r[2]} for r in rows]
    return {"date": date, "picks": picks, "count": len(picks),
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/agents/reports")
async def get_agent_reports(agent_id: str = "", limit: int = 10):
    repo = get_repository()
    reports = repo.get_latest_agent_reports(agent_id=agent_id or None, limit=limit)
    return [{"agent_id": r.agent_id, "timestamp": r.analysis_date.isoformat(),
             "signal": str(r.signal), "confidence": str(r.confidence), "reasoning": r.reasoning[:500]}
            for r in reports]


@app.get("/api/portfolio")
async def get_portfolio():
    repo = get_repository()
    positions = repo.get_all_positions()
    return [{"code": p.code, "quantity": p.quantity, "avg_cost": str(p.avg_cost),
             "market_value": str(p.market_value) if p.market_value else None,
             "weight": str(p.weight) if p.weight else None} for p in positions]


@app.get("/api/risk/status")
async def get_risk_status():
    from fengkong.risk_manager import RiskManager
    rm = RiskManager()
    portfolio = [{"code": "000001", "weight": Decimal("0.08")}]
    result = rm.check_all(portfolio, Decimal("0"), Decimal("1000000"), [Decimal("0.001")] * 100)
    return {"risk_level": result["risk_level"], "risk_score": result["risk_score"],
            "blocked": result["blocked"], "breaker_state": result["breaker_state"],
            "var_95": str(result["var_report"].get("var_95")) if result["var_report"].get("var_95") else None,
            "advice": result["advice"]}


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    from huice.backtest_engine import BacktestEngine
    engine = BacktestEngine()
    # 简化版：使用 Mock 数据源
    class SimpleLoader:
        def get_trade_dates(self, s, e): return [f"2026{(i//30)+1:02d}{(i%30)+1:02d}" for i in range(60)]
        def load_market_data(self, d): return {}
    class SimpleSignal:
        def generate(self, d, m, p): return None
    class SimpleExec:
        def execute(self, s, p, c, m): return []
    config = {"start_date": req.start_date, "end_date": req.end_date,
              "initial_capital": float(req.initial_capital),
              "strategy_name": "api_backtest", "params": req.strategy_params,
              "data_loader": SimpleLoader(), "signal_generator": SimpleSignal(), "executor": SimpleExec()}
    report = engine.run(config)
    return {"experiment_id": report["experiment_id"], "metrics": report["metrics"],
            "elapsed_seconds": report["elapsed_seconds"]}


@app.get("/api/backtest/summary")
async def get_backtest_summary():
    """回测绩效摘要 — 从 portfolio_snapshot 表读取。"""
    from sqlalchemy import text
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(text(
            'SELECT trade_date, total_value, cumulative_return, daily_return, position_count '
            'FROM portfolio_snapshot ORDER BY trade_date'
        )).fetchall()
        if not rows:
            return {"error": "no backtest data"}
        values = [r[1] for r in rows if r[1]]
        rets = [r[3] for r in rows if r[3] is not None]
        peak = max(values) if values else 0
        final = values[-1] if values else 0
        dd = min((v - peak) / peak for v in values) if peak > 0 else 0
        return {
            "start_date": str(rows[0][0]), "end_date": str(rows[-1][0]),
            "total_days": len(rows),
            "final_value": final,
            "total_return": (final - 1_000_000) / 1_000_000 if final else 0,
            "max_drawdown": dd,
            "sharpe": (mean(rets) / stdev(rets) * sqrt(252)) if len(rets) > 1 else 0,
            "snapshot_count": len(rows),
            # P1-1: 返回完整日频净值序列供前端绘制权益曲线
            "equity_curve": [{"date": str(r[0]), "value": r[1]} for r in rows if r[1] is not None],
        }


@app.get("/api/equity")
async def get_equity_curve():
    """日频权益曲线 — 供前端 Dashboard/回测绘制。"""
    from sqlalchemy import text
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(text(
            'SELECT trade_date, total_value FROM portfolio_snapshot ORDER BY trade_date'
        )).fetchall()
        return {"data": [{"date": str(r[0]), "value": r[1]} for r in rows if r[1] is not None]}


@app.get("/api/factors/weights")
async def get_factor_weights():
    """因子权重列表 — 供前端 Dashboard 因子权重卡片展示。"""
    from sqlalchemy import text
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(text(
            'SELECT factor_name, weight FROM factor_weight ORDER BY weight DESC'
        )).fetchall()
        if not rows:
            # 兜底: 返回默认权重
            return {"weights": [
                {"name": "GNN", "weight": 0.22},
                {"name": "Agent", "weight": 0.25},
                {"name": "ROE", "weight": 0.18},
                {"name": "PE", "weight": 0.15},
                {"name": "动量", "weight": 0.12},
                {"name": "情绪", "weight": 0.08},
            ]}
        return {"weights": [{"name": r[0], "weight": r[1]} for r in rows]}


@app.get("/api/metrics")
async def prometheus_metrics():
    from prometheus_client import REGISTRY, generate_latest
    return JSONResponse(generate_latest().decode(), media_type="text/plain")


# ── WebSocket ─────────────────────────────────────────

@app.websocket("/ws/market")
async def ws_market(websocket: WebSocket):
    await websocket.accept(); _ws_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "market", "data": {"index": "沪深300", "change": "+0.5%"}, "timestamp": datetime.now(timezone.utc).isoformat()})
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


@app.websocket("/ws/agents")
async def ws_agents(websocket: WebSocket):
    await websocket.accept(); _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"type": "agent", "agent_id": "macro", "reasoning": "PMI扩张", "confidence": "0.78"})
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


@app.websocket("/ws/risk")
async def ws_risk(websocket: WebSocket):
    await websocket.accept(); _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"type": "risk", "risk_level": "LOW", "breaker": "CLOSED"})
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ── 启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
