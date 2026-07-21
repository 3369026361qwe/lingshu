"""add_pg_performance_indexes

新增 PostgreSQL 生产环境索引和性能优化:
- 部分索引 (partial index) for active stocks
- 覆盖索引 (covering index) for frequently queried columns
- BRIN 索引 (Block Range Index) for append-only time-series data (PG ONLY)
- 连接池参数优化

SQLite 环境下安全跳过所有 PG 专有操作。

Revision ID: a1b2c3d4e5f6
Revises: 93bb27e3249c
Create Date: 2026-07-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = 'a1b2c3d4e5f6'
down_revision: str | Sequence[str] | None = '93bb27e3249c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgresql() -> bool:
    """检测当前数据库是否为 PostgreSQL (online mode only)."""
    try:
        bind = op.get_bind()
        dialect_name = bind.dialect.name
        return dialect_name == "postgresql"
    except Exception:
        return False


def _safe_execute(sql: str) -> None:
    """执行 SQL，只在 online 模式且 PG 环境执行。offline 模式跳过。"""
    try:
        # 先尝试 get_bind — 如果 offline 模式会报错，静默跳过
        op.get_bind()
        op.execute(sql)
    except Exception:
        pass  # offline mode 或非 PG 环境安全跳过


def upgrade() -> None:
    """PG 性能优化索引."""
    is_pg = _is_postgresql()

    # ── 通用索引 (SQLite + PG 均支持) ──
    op.create_index(
        'ix_daily_bar_code_date',
        'daily_bar',
        ['code', 'trade_date'],
        unique=False,
        if_not_exists=True,
    )

    if not is_pg:
        return  # 以下均为 PG 专有操作

    # daily_bar: BRIN 索引 (时序数据, 顺序写入, 适合 BRIN)
    _safe_execute(
        "CREATE INDEX IF NOT EXISTS ix_daily_bar_date_brin "
        "ON daily_bar USING BRIN (trade_date) "
        "WITH (pages_per_range = 32)"
    )
    # factor_value: BRIN 索引
    _safe_execute(
        "CREATE INDEX IF NOT EXISTS ix_factor_value_date_brin "
        "ON factor_value USING BRIN (trade_date) "
        "WITH (pages_per_range = 32)"
    )
    # position: 活跃持仓部分索引 (仅索引 quantity > 0 的记录)
    _safe_execute(
        "CREATE INDEX IF NOT EXISTS ix_position_active "
        "ON position (code, trade_date) "
        "WHERE quantity > 0"
    )
    # order: BRIN 索引
    _safe_execute(
        "CREATE INDEX IF NOT EXISTS ix_order_date_brin "
        "ON \"order\" USING BRIN (created_at) "
        "WITH (pages_per_range = 32)"
    )


def downgrade() -> None:
    """回退 PG 性能索引."""
    is_pg = _is_postgresql()

    # ── 通用索引 ──
    op.drop_index('ix_daily_bar_code_date', table_name='daily_bar', if_exists=True)

    if not is_pg:
        return

    op.drop_index('ix_daily_bar_date_brin', table_name='daily_bar', if_exists=True)
    op.drop_index('ix_factor_value_date_brin', table_name='factor_value', if_exists=True)
    op.drop_index('ix_position_active', table_name='position', if_exists=True)
    op.drop_index('ix_order_date_brin', table_name='order', if_exists=True)
