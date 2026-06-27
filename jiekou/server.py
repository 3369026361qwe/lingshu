"""FastAPI 入口 — 应用创建 + 路由注册 + 启动。"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, date
from decimal import Decimal
from statistics import mean

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from jiekou.dependencies import get_repository
from jiekou.routes.selection_routes import router as selection_router
from jiekou.routes.agent_routes import router as agent_router, ws_router as agent_ws_router
from jiekou.routes.portfolio_routes import router as portfolio_router
from jiekou.routes.risk_routes import router as risk_router, ws_router as risk_ws_router
from jiekou.routes.huice_routes import router as huice_router
from jiekou.routes.gnn_routes import router as gnn_router
from jiekou.routes.trade_routes import router as trade_router

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """应用生命周期：启动时预载市场缓存 + 日终流水线，关闭时清理。"""
    _ensure_market_task()
    _ensure_daily_task()
    yield


from jiekou.middleware import APIKeyMiddleware, RateLimitMiddleware

app = FastAPI(title="灵枢 LingShu API", version="3.0.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)
app.add_middleware(APIKeyMiddleware)

# 注册 REST 路由
app.include_router(selection_router)
app.include_router(agent_router)
app.include_router(portfolio_router)
app.include_router(risk_router)
app.include_router(huice_router)
app.include_router(gnn_router)
app.include_router(trade_router)

# 注册 WebSocket 路由（无 /api 前缀）
app.include_router(agent_ws_router)
app.include_router(risk_ws_router)


_market_cache: dict = {
    "latest_date": "",
    "stock_count": 0,
    "avg_change_pct": 0.0,
    "csi300": {"index": "沪深300", "value": "—", "change": "+0.00%"},
    "csi500": {"index": "中证500", "value": "—", "change": "+0.00%"},
    "chinext": {"index": "创业板指", "value": "—", "change": "+0.00%"},
    "updated_at": "",
}
_market_cache_lock = asyncio.Lock()
_MARKET_REFRESH_SEC = 60
_market_task: asyncio.Task | None = None


async def _refresh_market_cache() -> None:
    """从 DB 查询最新市场概览数据并更新缓存。

    通过股票代码前缀近似计算三大指数涨跌幅：
      - 沪深300: 60xxxx, 000xxx (沪市主板)
      - 中证500: 002xxx, 001xxx (深市主板/中小板)
      - 创业板指: 300xxx, 301xxx
    """
    from sqlalchemy import text

    def _compute_index(rows, code_prefixes: tuple) -> dict:
        """从 OHLCV 行计算指数的聚合涨跌幅。"""
        vals = []
        total_mv = 0.0
        for code, close, prev_close, volume in rows:
            try:
                if not any(str(code).startswith(p) for p in code_prefixes):
                    continue
                c, pc = float(str(close)), float(str(prev_close))
                if pc > 0 and c > 0:
                    chg = (c - pc) / pc * 100
                    mv = c * float(str(volume or 0))
                    vals.append((chg, mv))
                    total_mv += mv
            except (ValueError, ZeroDivisionError):
                pass

        if not vals:
            return {"value": "—", "change": "+0.00%", "up": True}

        if total_mv > 0:
            wgt_chg = sum(chg * mv for chg, mv in vals) / total_mv
        else:
            wgt_chg = sum(chg for chg, _ in vals) / len(vals)

        return {
            "value": f"{5000 + wgt_chg * 50:.2f}",  # 近似指数点位
            "change": f"{wgt_chg:+.2f}%",
            "up": wgt_chg >= 0,
        }

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
                    "SELECT a.code, a.close, b.close AS prev_close, a.volume "
                    "FROM daily_bar a "
                    "JOIN daily_bar b ON a.code = b.code AND b.trade_date = date(:d, '-1 day') "
                    "WHERE a.trade_date = :d"
                ),
                {"d": latest_date},
            ).fetchall()

            # 计算各指数
            csi300 = _compute_index(rows, ("60", "000"))
            csi500 = _compute_index(rows, ("002", "001"))
            chinext = _compute_index(rows, ("300", "301"))

            # 全市场平均涨跌幅
            changes = []
            for code, close, prev_close, _vol in rows:
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
                    "csi300": {"index": "沪深300", **csi300},
                    "csi500": {"index": "中证500", **csi500},
                    "chinext": {"index": "创业板指", **chinext},
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


# ── 日终调度器 ────────────────────────────────────────

_daily_task: asyncio.Task | None = None
_DAILY_RUN_HOUR = 15  # 15:00 收盘后
_DAILY_RUN_MINUTE = 30
_DAILY_ENABLED = os.getenv("LINGSHU_DAILY_AUTO", "0") == "1"


async def _daily_pipeline_loop():
    """日终自动流水线（需 LINGSHU_DAILY_AUTO=1 启用）。"""
    if not _DAILY_ENABLED:
        return
    while True:
        now = datetime.now(timezone.utc)
        # 北京时间 = UTC+8
        bj_hour = (now.hour + 8) % 24
        bj_minute = now.minute
        # 计算距离下一个 15:30 的秒数
        target_minutes = _DAILY_RUN_HOUR * 60 + _DAILY_RUN_MINUTE
        current_minutes = bj_hour * 60 + bj_minute
        wait_minutes = (target_minutes - current_minutes) % (24 * 60)
        if wait_minutes == 0:
            wait_minutes = 24 * 60  # 刚好到点，等一天
        wait_seconds = wait_minutes * 60
        _logger.info("Daily pipeline next run in %d min (BJ %02d:%02d)", wait_minutes, _DAILY_RUN_HOUR, _DAILY_RUN_MINUTE)
        await asyncio.sleep(wait_seconds)
        # 执行日终流水线
        try:
            _logger.info("Starting daily pipeline...")
            from scripts.run_daily_pipeline import main as run_pipeline
            await asyncio.to_thread(run_pipeline)
            _logger.info("Daily pipeline completed")
        except Exception as exc:
            _logger.error("Daily pipeline failed: %s", exc)


def _ensure_daily_task():
    global _daily_task
    if _DAILY_ENABLED and (_daily_task is None or _daily_task.done()):
        _daily_task = asyncio.ensure_future(_daily_pipeline_loop())


# ── 启动 ──────────────────────────────────────────────

# ── 前端静态文件 ────────────────────────────────────

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "qianduan" / "dist"


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """SPA 前端 — 必须在所有 API 路由之后注册。路径穿越已防护。"""
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        return JSONResponse({"error": "not found"}, 404)
    if _FRONTEND_DIST.exists():
        fp = (_FRONTEND_DIST / full_path).resolve()
        # 防止路径穿越 (../../.env 等)
        if fp.is_relative_to(_FRONTEND_DIST.resolve()) and fp.is_file() and "." in (fp.suffix or ""):
            return FileResponse(fp)
        # SPA 回退
        return HTMLResponse((_FRONTEND_DIST / "index.html").read_text(encoding="utf-8"))
    return HTMLResponse("<h3>前端未构建 — cd qianduan && npm run build</h3>", status_code=503)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
