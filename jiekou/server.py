"""FastAPI 入口 — 应用创建 + 路由注册 + 启动。"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, date
from decimal import Decimal
from statistics import mean

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from jiekou.dependencies import get_repository
from jiekou.routes.selection_routes import router as selection_router
from jiekou.routes.agent_routes import router as agent_router, ws_router as agent_ws_router
from jiekou.routes.portfolio_routes import router as portfolio_router
from jiekou.routes.risk_routes import router as risk_router, ws_router as risk_ws_router
from jiekou.routes.huice_routes import router as huice_router

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """应用生命周期：启动时预载市场缓存，关闭时清理。"""
    _ensure_market_task()
    yield


app = FastAPI(title="灵枢 LingShu API", version="3.0.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 注册 REST 路由
app.include_router(selection_router)
app.include_router(agent_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(huice_router)

# 注册 WebSocket 路由（无 /api 前缀）
app.include_router(agent_ws_router)
app.include_router(risk_ws_router)

# ── 市场数据缓存 ────────────────────────────────────────

_market_cache: dict = {
    "latest_date": "",
    "stock_count": 0,
    "avg_change_pct": 0.0,
    "updated_at": "",
}
_market_cache_lock = asyncio.Lock()
_MARKET_REFRESH_SEC = 60
_market_task: asyncio.Task | None = None


async def _refresh_market_cache() -> None:
    """从 DB 查询最新市场概览数据并更新缓存。"""
    from sqlalchemy import text
    try:
        repo = get_repository()
        with repo._session as s:
            latest_date = s.execute(
                text("SELECT MAX(trade_date) FROM daily_bar")
            ).scalar()
            if not latest_date:
                return

            latest_date = str(latest_date)
            rows = s.execute(
                text(
                    "SELECT a.close, b.close AS prev_close "
                    "FROM daily_bar a "
                    "JOIN daily_bar b ON a.code = b.code AND b.trade_date = date(:d, '-1 day') "
                    "WHERE a.trade_date = :d"
                ),
                {"d": latest_date},
            ).fetchall()

            changes = []
            for close, prev_close in rows:
                try:
                    c, pc = float(str(close)), float(str(prev_close))
                    if pc > 0:
                        changes.append((c - pc) / pc * 100)
                except (ValueError, ZeroDivisionError):
                    pass

            async with _market_cache_lock:
                _market_cache.update({
                    "latest_date": latest_date,
                    "stock_count": len(rows),
                    "avg_change_pct": round(mean(changes), 2) if changes else 0.0,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
    except Exception as exc:
        _logger.warning("Market cache refresh failed: %s", exc)


def _ensure_market_task():
    """确保市场刷新后台任务已启动（惰性，TestClient 安全）。"""
    global _market_task
    if _market_task is None or _market_task.done():
        _market_task = asyncio.ensure_future(_market_refresh_loop())


async def _market_refresh_loop():
    """后台循环刷新市场缓存。"""
    await _refresh_market_cache()  # 启动时立即刷新一次
    while True:
        await asyncio.sleep(_MARKET_REFRESH_SEC)
        await _refresh_market_cache()


# WebSocket 连接管理
_ws_clients: list[WebSocket] = []


# ── 通用端点 ──────────────────────────────────────────

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
    """个股基本信息。"""
    repo = get_repository()
    s = repo.get_stock_by_code(code)
    if not s:
        return JSONResponse({"error": "not found"}, 404)
    return {"code": s.code, "name": s.name, "exchange": s.exchange}


@app.get("/api/stocks/{code}/daily")
async def get_daily_bars(code: str, start: str = "20260101", end: str = "20260630"):
    """个股日线行情。"""
    repo = get_repository()
    bars = repo.get_daily_bars(
        code,
        date.fromisoformat(start[:4] + "-" + start[4:6] + "-" + start[6:]),
        date.fromisoformat(end[:4] + "-" + end[4:6] + "-" + end[6:]),
    )
    return [
        {
            "trade_date": str(b.trade_date),
            "open": str(b.open),
            "high": str(b.high),
            "low": str(b.low),
            "close": str(b.close),
            "volume": str(b.volume),
        }
        for b in bars
    ]


@app.get("/api/metrics")
async def prometheus_metrics():
    """Prometheus 指标导出。"""
    from prometheus_client import REGISTRY, generate_latest
    return JSONResponse(generate_latest().decode(), media_type="text/plain")


# ── WebSocket ─────────────────────────────────────────

@app.websocket("/ws/market")
async def ws_market(websocket: WebSocket):
    """市场行情实时推送 — 从 DB 缓存获取最新市场概览。"""
    _ensure_market_task()
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
            async with _market_cache_lock:
                data = dict(_market_cache)
            await websocket.send_json({
                "type": "market",
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ── 启动 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
