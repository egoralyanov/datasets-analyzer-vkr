"""add_file_hash_to_datasets

Revision ID: d75615d1ab2f
Revises: c15193c9b987
Create Date: 2026-05-07 17:43:21.561932

Спринт 6, Phase 4.1. Дедупликация загружаемых датасетов:
- Колонка `file_hash CHAR(64)` хранит SHA-256 содержимого файла.
- Backfill: проходим по существующим записям, для каждой считаем хэш от
  файла на диске. Если файла нет — записываем sentinel `missing:{id}`,
  чтобы не блокировать миграцию (sentinel длиной 8+36=44 ≤ 64 символов).
- Уникальный индекс `(user_id, file_hash)` обеспечивает дедупликацию в
  рамках одного пользователя; разные юзеры могут загружать один файл.
"""
import hashlib
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd75615d1ab2f'
down_revision: Union[str, None] = 'c15193c9b987'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_of_file(path: str) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            sha.update(chunk)
    return sha.hexdigest()


def upgrade() -> None:
    # 1. Колонка nullable — для backfill.
    op.add_column(
        "datasets",
        sa.Column("file_hash", sa.String(length=64), nullable=True),
    )

    # 2. Backfill существующих записей: SHA-256 от файла на диске или
    #    sentinel `missing:{id}` если файл потерян.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT id, storage_path FROM datasets WHERE file_hash IS NULL")
    ).fetchall()
    backfilled = 0
    sentineled = 0
    for row in rows:
        try:
            file_hash = _sha256_of_file(row.storage_path)
            backfilled += 1
        except (FileNotFoundError, OSError):
            file_hash = f"missing:{row.id}"
            sentineled += 1
        bind.execute(
            sa.text("UPDATE datasets SET file_hash = :h WHERE id = :id"),
            {"h": file_hash, "id": row.id},
        )
    print(
        f"[migration d75615d1ab2f] file_hash backfill: "
        f"{backfilled} hashed, {sentineled} sentinel"
    )

    # 3. Делаем NOT NULL.
    op.alter_column(
        "datasets",
        "file_hash",
        existing_type=sa.String(length=64),
        nullable=False,
    )

    # 4. Уникальный индекс: один и тот же файл — только один раз на пользователя.
    op.create_index(
        "ix_datasets_user_file_hash_unique",
        "datasets",
        ["user_id", "file_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_datasets_user_file_hash_unique", table_name="datasets"
    )
    op.drop_column("datasets", "file_hash")
