"""
ORM-модель записи каталога внешних датасетов.

Соответствует таблице `external_datasets` (см. .knowledge/architecture/database.md, раздел 8,
и .knowledge/methods/dataset-matching.md — алгоритм поиска похожих через pgvector).

Каталог хранит эталонные датасеты из открытых источников (sklearn, UCI, GitHub),
для каждой записи посчитан embedding на основе meta-features того же набора признаков,
что вычисляет наш профайлер для пользовательских датасетов. Поиск top-K похожих
выполняется в Слое сервисов (`dataset_matcher.py`) через pgvector-оператор
`<=>` (косинусное расстояние) с поддержкой HNSW-индекса.

Источник записей — `backend/ml/data/real_set.json` (см. `recommender-ml.md`,
раздел «Реальная часть выборки»). Синтетические датасеты в каталог не попадают:
у них нет осмысленного title/source_url для UI «Похожие датасеты».

Ключ идемпотентности — `title` (UNIQUE). Seed-скрипт делает
`INSERT ... ON CONFLICT (title) DO UPDATE SET ...`, что позволяет пересчитывать
embedding при изменении набора meta-features или версии scaler без удаления
записей и без дубликатов.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class ExternalDataset(Base):
    __tablename__ = "external_datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    title: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type_code: Mapped[str] = mapped_column(String(50), nullable=False)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    meta_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
