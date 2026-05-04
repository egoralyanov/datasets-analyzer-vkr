"""
API-эндпоинты админ-панели.

Контракт:
- GET /api/admin/stats        → 200 + AdminStats
- GET /api/admin/users?page=&size=  → 200 + AdminUserListResponse

Оба под `Depends(get_current_admin)` — non-admin пользователи получают 403,
что HTTP-семантически корректно: пользователь известен, но прав не хватает.
В отличие от 404 в других местах (где «не палим существование») здесь
известно, что эндпоинты админские, и 403 — стандартное поведение.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.models.user import User
from app.repositories import admin_repo
from app.schemas.admin import (
    AdminStats,
    AdminUserListItem,
    AdminUserListResponse,
)

router = APIRouter(tags=["admin"])


@router.get("/admin/stats", response_model=AdminStats)
def get_stats(
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> AdminStats:
    """Сводная статистика: счётчики и success-rate."""
    return AdminStats(**admin_repo.compute_admin_stats(db))


@router.get("/admin/users", response_model=AdminUserListResponse)
def list_admin_users(
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    size: int = Query(20, ge=1, le=100, description="Размер страницы (1..100)"),
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> AdminUserListResponse:
    """
    Пагинированный список всех пользователей системы. Сортировка
    `created_at DESC`. Для каждой строки подгружено число датасетов и
    анализов через scalar-subquery.
    """
    rows, total = admin_repo.list_users_paginated(db, page=page, size=size)
    # max(1, ...) — защита от «Стр 1 из 0», когда total=0 и UI без user'ов.
    # На реальной admin-странице current_admin сам там всегда есть, так что
    # total >= 1; защита бесплатная для пустых тестовых стендов.
    pages = max(1, (total + size - 1) // size) if size else 1
    return AdminUserListResponse(
        items=[
            AdminUserListItem(
                id=user.id,
                email=user.email,
                username=user.username,
                role=user.role,
                created_at=user.created_at,
                datasets_count=dc,
                analyses_count=ac,
            )
            for user, dc, ac in rows
        ],
        total=total,
        page=page,
        size=size,
        pages=pages,
    )
