"""
ORM-модель результата анализа датасета.

Соответствует таблице `analysis_results` (см. .knowledge/architecture/database.md, раздел 4).
Связана с `analyses` отношением 1:1 — `analysis_id` одновременно PK и FK.

Поля:
- `meta_features` JSONB — все ~30 признаков профиля (структурные, статистические,
  целевые, корреляции, выбросы, сэмплирование). Также сюда вкладываются
  `distributions` и `correlations` для отрисовки графиков на фронте — они часть
  профиля, выделять их в отдельные колонки смысла нет.
- `embedding` vector(128) — нормализованный вектор meta-features для подбора
  похожих датасетов через косинусную меру (Спринт 3). На Спринте 2 заполняется null.
- `task_recommendation` JSONB — рекомендация типа задачи (Спринт 3).
- `baseline` JSONB — метрики baseline-моделей (Спринт 3).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        primary_key=True,
    )
    meta_features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    task_recommendation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    baseline: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="result")
