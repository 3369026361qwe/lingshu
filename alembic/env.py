"""Alembic 环境配置 — 自动从 shujuku.models.Base 读取模型元数据。"""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# 确保项目根目录在 sys.path 中
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# 加载 .env（如果有）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(_proj_root) / ".env")
except ImportError:
    pass

# 读取 Alembic 配置
config = context.config

# 日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 从环境变量覆盖数据库 URL（生产环境用环境变量注入）
_db_url = os.getenv("DATABASE_URL", "").strip()
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

# 导入所有模型 → 注册到 Base.metadata
import shujuku.models.fengkong_models  # noqa: E402, F401
import shujuku.models.jiaoyi_models  # noqa: E402, F401
import shujuku.models.juece_models  # noqa: E402, F401
import shujuku.models.market_models  # noqa: E402, F401
import shujuku.models.yinzi_models  # noqa: E402, F401
import shujuku.models.zhinengti_models  # noqa: E402, F401
from shujuku.models import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线迁移 — 生成 SQL 而非直接执行。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线迁移 — 直接对数据库执行。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
