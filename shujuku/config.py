"""
数据库配置。

读取环境变量构造数据库连接 URL，支持 SQLite (默认) 和 PostgreSQL。
Redis URL 也在此集中管理，不可用时返回 None (触发优雅降级)。

环境变量:
    DATABASE_URL  — 完整数据库连接串 (优先级最高)
    DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD — PostgreSQL 分段配置
    REDIS_URL     — Redis 连接串
    DB_ECHO       — 是否打印 SQL 日志 (默认 0)
"""

import os
import urllib.parse

# ── Database URL ────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if _DATABASE_URL:
    DATABASE_URL: str = _DATABASE_URL
else:
    # 尝试从分段环境变量构造 PostgreSQL URL
    _host = os.getenv("DB_HOST", "localhost")
    _port = os.getenv("DB_PORT", "5432")
    _name = os.getenv("DB_NAME", "lingshu")
    _user = os.getenv("DB_USER", "lingshu")
    _pwd = os.getenv("DB_PASSWORD", "lingshu")

    if os.getenv("DB_HOST"):
        # PostgreSQL 模式
        _pwd_encoded = urllib.parse.quote_plus(_pwd)
        DATABASE_URL = f"postgresql://{_user}:{_pwd_encoded}@{_host}:{_port}/{_name}"
    else:
        # 默认: SQLite 开发模式
        _db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lingshu.db")
        DATABASE_URL = f"sqlite:///{_db_path}"

# ── Redis URL ───────────────────────────────────────────────
REDIS_URL: str | None = os.getenv("REDIS_URL", "").strip() or None

# ── SQL Echo ────────────────────────────────────────────────
DB_ECHO: bool = os.getenv("DB_ECHO", "0") == "1"


def is_sqlite() -> bool:
    """判断当前是否为 SQLite 模式。"""
    return DATABASE_URL.startswith("sqlite")


def is_postgresql() -> bool:
    """判断当前是否为 PostgreSQL 模式。"""
    return DATABASE_URL.startswith("postgresql")
