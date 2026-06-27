"""initial schema

基线迁移 — 所有表由 Base.metadata.create_all() 创建。
此迁移仅调整 fusion_score 索引以匹配 ORM 模型定义。

Revision ID: 64712d4d982f
Revises:
Create Date: 2026-06-28 01:50:25.996877
"""
from typing import Sequence, Union
from alembic import op

revision: str = '64712d4d982f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """初始 schema — 索引调整。表结构由 ORM 创建，无需变更。"""
    # fusion_score 索引更新：匹配 juece_models.py 定义
    try:
        op.drop_index('ix_fusion_score_code', table_name='fusion_score')
    except Exception:
        pass
    try:
        op.drop_index('ix_fusion_score_date', table_name='fusion_score')
    except Exception:
        pass
    op.create_index('ix_fusion_date_code', 'fusion_score', ['trade_date', 'code'], unique=True, if_not_exists=True)
    op.create_index('ix_fusion_date_rank', 'fusion_score', ['trade_date', 'rank'], if_not_exists=True)
    op.create_index('ix_fusion_score_trade_date', 'fusion_score', ['trade_date'], if_not_exists=True)


def downgrade() -> None:
    """回退索引到旧状态。"""
    op.drop_index('ix_fusion_score_trade_date', table_name='fusion_score', if_exists=True)
    op.drop_index('ix_fusion_date_rank', table_name='fusion_score', if_exists=True)
    op.drop_index('ix_fusion_date_code', table_name='fusion_score', if_exists=True)
    op.create_index('ix_fusion_score_code', 'fusion_score', ['code'], if_not_exists=True)
    op.create_index('ix_fusion_score_date', 'fusion_score', ['trade_date'], if_not_exists=True)
