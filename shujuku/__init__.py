"""
shujuku — 持久化层 (数据库层)

提供 SQLAlchemy ORM 模型、CRUD 仓库、Redis 缓存、Prometheus 指标。
支持 SQLite (开发) / PostgreSQL (生产) 双模式，Redis 可选优雅降级。

Usage:
    from shujuku.config import DATABASE_URL
    from shujuku.session import get_session
    from shujuku.models import Base, StockInfo, DailyBar
    from shujuku.repository import Repository
    from shujuku.redis_cache import CacheManager
"""

from shujuku.config import DATABASE_URL, DB_ECHO, REDIS_URL
from shujuku.session import SessionContext, get_session, init_db

__all__ = [
    "DATABASE_URL",
    "REDIS_URL",
    "DB_ECHO",
    "get_session",
    "SessionContext",
    "init_db",
]
