"""
SQLAlchemy 会话管理。

提供线程安全的 Session 工厂和上下文管理器。
SQLite 模式下自动启用 WAL 模式和跨线程支持。
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from shujuku.config import DATABASE_URL, DB_ECHO, get_pg_pool_config, is_sqlite
from shujuku.metrics import session_pool_size

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

# ── Engine ──────────────────────────────────────────────────
_connect_args: dict = {}
if is_sqlite():
    _connect_args["check_same_thread"] = False

# H1: SQLite 使用 NullPool 禁用连接池
_engine_kwargs: dict = {
    "echo": DB_ECHO,
    "connect_args": _connect_args,
}
if is_sqlite():
    _engine_kwargs["poolclass"] = NullPool
else:
    pool_cfg = get_pg_pool_config()
    _engine_kwargs["pool_size"] = pool_cfg["pool_size"]
    _engine_kwargs["max_overflow"] = pool_cfg["max_overflow"]
    _engine_kwargs["pool_timeout"] = pool_cfg["pool_timeout"]
    _engine_kwargs["pool_recycle"] = pool_cfg["pool_recycle"]
    _engine_kwargs["pool_pre_ping"] = pool_cfg["pool_pre_ping"]

_engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs)

# ── SQLite WAL ──────────────────────────────────────────────
if is_sqlite():

    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, *_):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ── Session Factory ─────────────────────────────────────────
_SessionLocal = sessionmaker(
    bind=_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_session() -> Session:
    """获取一个新的数据库会话。

    Returns:
        SQLAlchemy Session 对象。调用方负责关闭。
    """
    session = _SessionLocal()
    # 上报连接池状态
    try:
        pool = _engine.pool
        session_pool_size.labels(state="checked_out").set(pool.checkedout())
        session_pool_size.labels(state="overflow").set(pool.overflow() if hasattr(pool, 'overflow') else 0)
    except Exception as exc:
        logging.getLogger(__name__).debug("Failed to report pool metrics: %s", exc)
    return session


@contextmanager
def SessionContext() -> Generator[Session, None, None]:
    """会话上下文管理器，自动 commit/rollback + close。

    Usage:
        with SessionContext() as session:
            stock = session.get(StockInfo, "000001")
    """
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(drop_all: bool = False) -> None:
    """初始化数据库表结构。

    应在所有模型定义完成后调用一次。
    生产环境不应使用 drop_all=True。

    Args:
        drop_all: 是否先删除所有表（仅用于测试环境）

    Raises:
        RuntimeError: 生产环境下 drop_all=True 时抛出
    """
    # 延迟导入避免循环
    from shujuku.models import Base  # deferred import to avoid circular

    # 确保 data 目录存在 (SQLite)
    if is_sqlite():
        db_path = DATABASE_URL.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if drop_all:
        # ── 方案 A: 生产环境保护 ──────────────────────────
        env = os.getenv("LINGSHU_ENV", "production")  # 默认 production，安全优先
        if env != "dev":
            raise RuntimeError(
                "⛔ DANGER: init_db(drop_all=True) called!\n"
                f"   Current LINGSHU_ENV={env} (only 'dev' allows drop_all)\n"
                "   Set LINGSHU_ENV=dev to enable destructive operations.\n"
                "   This protection exists to prevent accidental data loss."
            )

        # ── 方案 B: 仅允许 test 数据库 ──────────────────
        if "test" not in DATABASE_URL.lower():
            raise RuntimeError(
                "⛔ REFUSED: init_db(drop_all=True) on non-test database!\n"
                f"   DATABASE_URL={DATABASE_URL}\n"
                "   drop_all is only allowed on databases with 'test' in the URL.\n"
                "   Set DATABASE_URL=sqlite:///./data/test_lingshu.db for testing."
            )

        # ── 方案 C: 警告日志 + 堆栈追踪 ──────────────────
        import traceback
        _logger.warning(
            "🗑️  DANGER: Dropping ALL database tables!\n"
            "   This will DELETE ALL DATA permanently.\n"
            "   Caller stack trace:\n%s",
            ''.join(traceback.format_stack())
        )

        Base.metadata.drop_all(_engine)
        _logger.warning("⚠️  All tables dropped. Database is now empty.")

    Base.metadata.create_all(_engine)
    _logger.info("Database tables created (drop_all=%s)", drop_all)
