"""analyses keep after dataset delete

Revision ID: a3f0c9d2e811
Revises: d75615d1ab2f
Create Date: 2026-05-20 12:00:00.000000

Удаление датасета больше не уносит связанные анализы и отчёты.

Что меняем:
1. Добавляем в `analyses` денормализованный снапшот метаданных датасета
   (`dataset_filename`, `dataset_format`, `dataset_n_rows`, `dataset_n_cols`,
   `dataset_file_size_bytes`). Имя файла нужно странице истории и PDF-отчёту,
   а после удаления Dataset брать его уже неоткуда.
2. Бэкфилим снапшот из текущих `datasets` для существующих анализов.
3. Делаем `analyses.dataset_id` nullable и пересоздаём FK с ON DELETE SET NULL
   вместо ON DELETE CASCADE — теперь удаление Dataset обнуляет ссылку, а
   анализ остаётся в истории.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f0c9d2e811"
down_revision: Union[str, None] = "d75615d1ab2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Снапшот-колонки. Все nullable: для уже удалённых датасетов в прошлом
    #    данных нет, и для будущих NULL-dataset_id они могут быть пустыми.
    op.add_column(
        "analyses",
        sa.Column("dataset_filename", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "analyses",
        sa.Column("dataset_format", sa.String(length=10), nullable=True),
    )
    op.add_column(
        "analyses",
        sa.Column("dataset_n_rows", sa.Integer(), nullable=True),
    )
    op.add_column(
        "analyses",
        sa.Column("dataset_n_cols", sa.Integer(), nullable=True),
    )
    op.add_column(
        "analyses",
        sa.Column("dataset_file_size_bytes", sa.BigInteger(), nullable=True),
    )

    # 2. Бэкфилл из текущих datasets. UPDATE … FROM — один SQL без Python-цикла.
    op.execute(
        """
        UPDATE analyses a
        SET
            dataset_filename = d.original_filename,
            dataset_format = d.format,
            dataset_n_rows = d.n_rows,
            dataset_n_cols = d.n_cols,
            dataset_file_size_bytes = d.file_size_bytes
        FROM datasets d
        WHERE a.dataset_id = d.id
        """
    )

    # 3. Сначала разрешаем NULL для dataset_id, потом пересоздаём FK с SET NULL.
    op.alter_column(
        "analyses",
        "dataset_id",
        existing_type=sa.dialects.postgresql.UUID(),
        nullable=True,
    )
    op.drop_constraint(
        "analyses_dataset_id_fkey", "analyses", type_="foreignkey"
    )
    op.create_foreign_key(
        "analyses_dataset_id_fkey",
        "analyses",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # Возвращаем CASCADE — но даунгрейд возможен только если все анализы
    # ссылаются на существующий датасет (orphan-анализы с dataset_id IS NULL
    # надо предварительно удалить или восстановить связь вручную).
    op.drop_constraint(
        "analyses_dataset_id_fkey", "analyses", type_="foreignkey"
    )
    op.create_foreign_key(
        "analyses_dataset_id_fkey",
        "analyses",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.alter_column(
        "analyses",
        "dataset_id",
        existing_type=sa.dialects.postgresql.UUID(),
        nullable=False,
    )

    op.drop_column("analyses", "dataset_file_size_bytes")
    op.drop_column("analyses", "dataset_n_cols")
    op.drop_column("analyses", "dataset_n_rows")
    op.drop_column("analyses", "dataset_format")
    op.drop_column("analyses", "dataset_filename")
