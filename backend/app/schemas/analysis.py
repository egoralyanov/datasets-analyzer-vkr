"""Pydantic-схемы для API анализа."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StartAnalysisRequest(BaseModel):
    """Параметры запуска анализа.

    `target_column` опционален: без него анализ выполняется без target (кластеризация),
    тип задачи определяется рекомендателем (Слой 1 правил + Слой 2 ML).
    """

    target_column: str | None = Field(
        default=None,
        description="Имя целевого столбца. Если не указано — анализ без target.",
    )


class AnalysisResponse(BaseModel):
    """Лёгкая модель статуса анализа — для polling и страницы истории."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    target_column: str | None
    started_at: datetime
    finished_at: datetime | None
    error_message: str | None


class QualityFlagResponse(BaseModel):
    """Один флаг качества для отрисовки на странице результата."""

    rule_code: str
    severity: str
    rule_name: str
    message: str
    context: dict[str, Any] | None


class AnalysisListItem(BaseModel):
    """Облегчённая запись для страницы «История анализов».

    Не путать с `AnalysisResponse` — там более полная модель для polling
    одного анализа. Здесь только то, что нужно для строки списка:
    идентификаторы, имя файла датасета (через joinedload), статус,
    рекомендованный тип задачи (если уже посчитан) и временные метки.
    """

    id: uuid.UUID
    dataset_id: uuid.UUID
    dataset_name: str
    status: str
    target_column: str | None
    recommended_task_type: str | None
    started_at: datetime
    finished_at: datetime | None


class AnalysisListResponse(BaseModel):
    """Пагинированный ответ GET /api/analyses."""

    items: list[AnalysisListItem]
    total: int
    page: int
    size: int
    pages: int


class AnalysisResultResponse(BaseModel):
    """Полный результат анализа: meta-features + флаги + рекомендация + embedding.

    `task_recommendation` и `embedding` опциональны (None) — они появляются
    в Спринте 3 и могут быть None для старых анализов из Спринта 2 либо когда
    рекомендатер/scaler отсутствуют (graceful degradation).
    """

    analysis: AnalysisResponse
    meta_features: dict[str, Any]
    flags: list[QualityFlagResponse]
    task_recommendation: dict[str, Any] | None = None
    embedding: list[float] | None = None
