"""enable_pgvector

Активация расширения pgvector в БД. Образ `pgvector/pgvector:pg15` уже содержит
библиотеку расширения, но её нужно явно подключить через CREATE EXTENSION —
autogenerate Alembic этого не делает, поэтому миграция написана вручную.

Расширение vector нужно для типа VECTOR(N), используемого в таблицах
analysis_results.embedding и external_datasets.embedding (создаются в Спринтах 2-3).

Revision ID: 0001
Revises:
Create Date: 2026-05-01

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector;")
