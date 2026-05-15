"""
ORM-модель загруженного пользователем датасета.

Соответствует таблице `datasets` (см. .project_docs/architecture/database.md, раздел 2).
В БД хранятся только метаданные и `storage_path` — сам файл лежит на диске
по пути /data/datasets/{user_id}/{uuid}.{ext}.

`file_hash` (Спринт 6, Phase 4.1) — SHA-256 содержимого файла, считается на
лету при сохранении. Уникальный составной индекс (user_id, file_hash)
обеспечивает дедупликацию: один и тот же файл не может быть загружен
одним пользователем дважды (вернётся 409). Разные пользователи могут
загружать один файл независимо — индекс scoped по user_id.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.user import User


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        Index(
            "ix_datasets_user_file_hash_unique",
            "user_id",
            "file_hash",
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    n_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_cols: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    user: Mapped["User"] = relationship(back_populates="datasets")
    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
