"""Pydantic-схемы для API PDF-отчётов."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


ReportStatus = Literal["pending", "running", "success", "failed"]


class ReportCreateResponse(BaseModel):
    """Ответ на POST /api/analyses/{id}/report (202 Accepted)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: ReportStatus


class ReportRead(BaseModel):
    """Ответ на GET /api/reports/{id}.

    user_id наружу не отдаём — клиент и так аутентифицирован под этим
    пользователем, дублирующая информация не нужна, плюс защита от
    случайной утечки чужого user_id при ошибке скоупа.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_id: uuid.UUID
    status: ReportStatus
    file_size_bytes: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class ReportConflictResponse(BaseModel):
    """
    Тело ответа 409 при конфликте создания отчёта.

    Поле `reason` — машинно-читаемое: фронт по нему различает два кейса:
    - `analysis_not_done` — анализ ещё не завершён (показать toast и
      попросить дождаться);
    - `report_in_progress` — отчёт уже генерируется, можно подключиться
      к polling существующего report_id.
    """

    detail: str
    reason: Literal["analysis_not_done", "report_in_progress"]
    report_id: uuid.UUID | None = None
    status: ReportStatus | None = None
