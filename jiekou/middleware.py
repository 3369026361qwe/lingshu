"""中间件 — API Key 鉴权 + 限流 + 日志。"""
import logging
import os
import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_logger = logging.getLogger(__name__)

# 读取 API Key（如未设置则鉴权功能关闭）
_API_KEY = os.getenv("LINGSHU_API_KEY", "").strip()
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# 读端点白名单（无 API Key 也能访问）
_READ_PATHS = {
    "/api/health", "/api/metrics", "/api/stocks", "/api/selection",
    "/api/agents/reports", "/api/portfolio", "/api/equity",
    "/api/risk/status", "/api/backtest/summary", "/api/factors/weights",
    "/api/gnn/graph", "/ws/market", "/ws/agents", "/ws/risk",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件。

    读端点（GET / 无 Key）：放行
    写端点（POST/PUT/DELETE / WebSocket send）：需 X-API-Key header 匹配

    环境变量 LINGSHU_API_KEY 为空时鉴权关闭，所有端点放行。
    """

    async def dispatch(self, request: Request, call_next):
        # 鉴权关闭
        if not _API_KEY:
            return await call_next(request)

        # WebSocket 的鉴权在连接时检查
        if request.url.path.startswith("/ws/"):
            return await call_next(request)

        # 读端点放行
        if request.url.path in _READ_PATHS or request.method == "GET":
            return await call_next(request)

        # 写端点检查 API Key
        if request.method in _WRITE_METHODS:
            api_key = request.headers.get("X-API-Key", "")
            if api_key != _API_KEY:
                _logger.warning("Unauthorized %s %s from %s",
                                request.method, request.url.path,
                                request.client.host if request.client else "unknown")
                return JSONResponse(
                    {"error": "unauthorized", "detail": "Missing or invalid X-API-Key header"},
                    status_code=401,
                )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """简单内存限流 — 每 IP 每分钟最多 60 次请求。"""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self._max = max_requests
        self._window = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        # 清理过期记录
        self._counters[ip] = [t for t in self._counters[ip] if now - t < self._window]

        if len(self._counters[ip]) >= self._max:
            _logger.warning("Rate limit exceeded for %s (%d requests/%ds)",
                            ip, len(self._counters[ip]), self._window)
            return JSONResponse(
                {"error": "rate_limited", "detail": f"Max {self._max} requests per {self._window}s"},
                status_code=429,
            )

        self._counters[ip].append(now)
        return await call_next(request)
