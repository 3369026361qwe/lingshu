"""
PostgreSQL 迁移脚本 (Alembic).

提供版本化的数据库架构迁移, 支持:
    - SQLite (开发模式) — 自动使用
    - PostgreSQL (生产模式) — 当 DB_HOST 设置时启用
    - 迁移历史追踪 (alembic_version 表)
    - 自动生成迁移脚本 (--autogenerate)

Usage:
    # 创建新迁移
    alembic revision --autogenerate -m "add_new_table"

    # 升级到最新
    alembic upgrade head

    # 回滚一个版本
    alembic downgrade -1

    # 查看当前版本
    alembic current
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 迁移目录 (alembic init 时需要)
# ══════════════════════════════════════════════════════════════

ALEMBIC_DIR = Path(__file__).parent / "alembic"
VERSIONS_DIR = ALEMBIC_DIR / "versions"

# 确保目录存在
ALEMBIC_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
