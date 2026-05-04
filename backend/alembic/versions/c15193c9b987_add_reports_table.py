"""add_reports_table

Revision ID: c15193c9b987
Revises: 565cc4d66be6
Create Date: 2026-05-04 13:00:26.438570

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c15193c9b987'
down_revision: Union[str, None] = '565cc4d66be6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'reports',
        sa.Column(
            'id',
            sa.UUID(),
            server_default=sa.text('gen_random_uuid()'),
            nullable=False,
        ),
        sa.Column('analysis_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column(
            'status',
            sa.String(length=20),
            server_default='pending',
            nullable=False,
        ),
        sa.Column('file_path', sa.String(length=1000), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('error', sa.String(length=1000), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_reports_analysis_id'),
        'reports',
        ['analysis_id'],
        unique=False,
    )
    op.create_index(
        'ix_reports_user_status',
        'reports',
        ['user_id', 'status'],
        unique=False,
    )
    # Внимание: автогенерация Alembic пытается дропнуть HNSW-индекс
    # external_datasets_embedding_idx, потому что не распознаёт его как
    # известный тип индекса. Удаление вручную убрано — индекс остаётся
    # на месте. Та же грабля закомментирована в миграции 565cc4d66be6.


def downgrade() -> None:
    op.drop_index('ix_reports_user_status', table_name='reports')
    op.drop_index(op.f('ix_reports_analysis_id'), table_name='reports')
    op.drop_table('reports')
