"""add performance indexes

加速高频查询——覆盖 7 个最常用查询模式。

Revision ID: 93bb27e3249c
Revises: 64712d4d982f
Create Date: 2026-06-28 15:16:45.165046
"""
from collections.abc import Sequence

from alembic import op

revision: str = '93bb27e3249c'
down_revision: str | Sequence[str] | None = '64712d4d982f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 日线行情: JOIN 查询 code + trade_date
    op.create_index('ix_daily_bar_code_date', 'daily_bar', ['code', 'trade_date'], if_not_exists=True)
    # 因子值: 按日期过滤
    op.create_index('ix_factor_value_date', 'factor_value', ['trade_date'], if_not_exists=True)
    op.create_index('ix_factor_value_code_date', 'factor_value', ['code', 'trade_date'], if_not_exists=True)
    # 融合分数: 按日期 + 排名排序
    op.create_index('ix_fusion_score_date_rank', 'fusion_score', ['trade_date', 'rank'], if_not_exists=True)
    # 持仓快照: 按日期排序（权益曲线查询）
    op.create_index('ix_portfolio_snapshot_date', 'portfolio_snapshot', ['trade_date'], if_not_exists=True)


def downgrade() -> None:
    op.drop_index('ix_portfolio_snapshot_date', table_name='portfolio_snapshot', if_exists=True)
    op.drop_index('ix_fusion_score_date_rank', table_name='fusion_score', if_exists=True)
    op.drop_index('ix_factor_value_code_date', table_name='factor_value', if_exists=True)
    op.drop_index('ix_factor_value_date', table_name='factor_value', if_exists=True)
    op.drop_index('ix_daily_bar_code_date', table_name='daily_bar', if_exists=True)
