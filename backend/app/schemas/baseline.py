"""Pydantic-схемы для baseline-обучения и подбора похожих датасетов."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class BaselineStartResponse(BaseModel):
    """Ответ на POST /api/analyses/{id}/baseline.

    Status_code эндпоинта зависит от состояния:
    - 202 Accepted — задача поставлена в очередь, baseline_status='running';
    - 200 OK — baseline уже обучен ранее, baseline_status='done' (идемпотентность).
    """

    analysis_id: uuid.UUID
    baseline_status: Literal["running", "done"]


class BaselineResultResponse(BaseModel):
    """Ответ на GET /api/analyses/{id}/baseline.

    Содержит текущее состояние и (если done) — обученные модели и метрики
    в формате контракта baseline-training.md.
    """

    baseline_status: str
    baseline: dict[str, Any] | None = None
    baseline_error: str | None = None


class SimilarDatasetResponse(BaseModel):
    """Один похожий датасет для UI-карточки SimilarDatasetsCard (Phase 7)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    source: str
    source_url: str | None
    task_type_code: str
    distance: float
