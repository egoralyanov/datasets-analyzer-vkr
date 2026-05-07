"""Pydantic-схемы админ-панели."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class AdminStats(BaseModel):
    """
    Сводка по системе для дашборда админки.

    `*_success_rate` — float в диапазоне [0, 1]; None если соответствующий
    `total_*` равен нулю (избегаем деления на ноль). Фронт показывает «—»
    при None и `value * 100`% иначе.
    """

    total_users: int
    total_datasets: int
    total_analyses: int
    total_reports: int
    analyses_success_rate: float | None
    reports_success_rate: float | None


class AdminUserListItem(BaseModel):
    """Одна строка таблицы пользователей в админке."""

    id: uuid.UUID
    email: str
    username: str
    role: str
    created_at: datetime
    datasets_count: int
    analyses_count: int


class AdminUserListResponse(BaseModel):
    """Пагинированный ответ GET /api/admin/users."""

    items: list[AdminUserListItem]
    total: int
    page: int
    size: int
    pages: int


class AdminUserDetail(BaseModel):
    """
    Детальная карточка пользователя для модалки на админ-странице
    (Спринт 6, Phase 4.5). Возвращается из GET /api/admin/users/{id}.

    `reports_count` считает только PDF-отчёты со status='success' — та же
    семантика, что в /api/datasets/{id}/usage (Phase 4.2): фронту
    интересны готовые артефакты, failed/pending показывать незачем.
    """

    id: uuid.UUID
    email: str
    username: str
    role: str
    created_at: datetime
    datasets_count: int
    analyses_count: int
    reports_count: int
