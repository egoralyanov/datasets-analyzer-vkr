"""Pydantic-схемы для API анализа."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StartAnalysisRequest(BaseModel):
    """Параметры запуска анализа: оба поля опциональные."""

    target_column: str | None = Field(
        default=None,
        description="Имя целевого столбца. Если не указано — анализ без target (кластеризация).",
    )
    hinted_task_type: str | None = Field(
        default=None,
        description="Подсказка от пользователя по типу задачи (используется в Спринте 3).",
    )


class AnalysisResponse(BaseModel):
    """Лёгкая модель статуса анализа — для polling и страницы истории."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    target_column: str | None
    hinted_task_type: str | None
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


class AnalysisResultResponse(BaseModel):
    """Полный результат анализа: meta-features + флаги."""

    analysis: AnalysisResponse
    meta_features: dict[str, Any]
    flags: list[QualityFlagResponse]
