"""
API-эндпоинты админ-панели.

Контракт:
- GET    /api/admin/stats             → 200 + AdminStats
- GET    /api/admin/users?page=&size= → 200 + AdminUserListResponse
- GET    /api/admin/users/{id}        → 200 + AdminUserDetail (Спринт 6, Phase 4.5)
- DELETE /api/admin/users/{id}        → 204 (Спринт 6, Phase 4.4)

Все эндпоинты под `Depends(get_current_admin)` — non-admin пользователи
получают 403, что HTTP-семантически корректно: пользователь известен, но
прав не хватает. В отличие от 404 в других местах (где «не палим
существование») здесь известно, что эндпоинты админские, и 403 —
стандартное поведение.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.file_storage import delete_dataset_file, delete_report_file
from app.models.user import User
from app.repositories import admin_repo, user_repo
from app.schemas.admin import (
    AdminStats,
    AdminUserDetail,
    AdminUserListItem,
    AdminUserListResponse,
)

logger = logging.getLogger(__name__)

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


@router.get("/admin/users/{user_id}", response_model=AdminUserDetail)
def get_admin_user_detail(
    user_id: uuid.UUID,
    _admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> AdminUserDetail:
    """
    Детали одного пользователя для модалки в админ-панели (Спринт 6,
    Phase 4.5). Включает агрегаты по датасетам, анализам и
    success-отчётам.

    Доступ: только role=admin (через `get_current_admin`). Не-admin →
    403, без auth → 401, несуществующий user_id → 404.
    """
    target = user_repo.get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    datasets_count, analyses_count, reports_count = (
        admin_repo.compute_user_aggregates(db, target.id)
    )
    return AdminUserDetail(
        id=target.id,
        email=target.email,
        username=target.username,
        role=target.role,
        created_at=target.created_at,
        datasets_count=datasets_count,
        analyses_count=analyses_count,
        reports_count=reports_count,
    )


@router.delete("/admin/users/{user_id}")
def delete_admin_user(
    user_id: uuid.UUID,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Удаляет пользователя со всем его контентом (Спринт 6, Phase 4.4).

    Авторизация: только role=admin (через `get_current_admin`). Не-admin
    получает 403; без auth — 401. Несуществующий user_id → 404.

    Бизнес-проверки идут в строгом порядке:
    1. user not found → 404
    2. self-удаление (`user_id == current_admin.id`) → 409 с плоским
       `{"detail": "Cannot delete your own admin account"}`. Безопасный
       default: admin не должен случайно отстрелить себе доступ.
    3. defense-in-depth: удаляемый — admin И в системе он единственный
       admin → 409 `{"detail": "Cannot delete the last admin account"}`.
       Этот кейс через self-check выше технически недостижим (для
       отправки DELETE нужен сам admin → он не последний), но проверка
       остаётся на случай будущих CLI-сценариев или второго пути.

    Каскад: FK ondelete=CASCADE из моделей унесёт datasets → analyses →
    analysis_results / quality_flags / reports. Файлы (storage_path
    датасетов и file_path PDF success-отчётов) собираем ДО
    `db.delete(user)`, после `db.commit()` — `unlink(missing_ok=True)`.
    Сбой физического удаления НЕ откатывает БД (записи уже консистентно
    удалены), а пишется WARNING со списком осиротевших путей.
    """
    target = user_repo.get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )

    if target.id == current_admin.id:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Cannot delete your own admin account"},
        )

    # Defense-in-depth: при self-check выше эта ветка обычно недостижима
    # (последний admin = только сам); оставляем на случай других путей
    # удаления (CLI-скрипт, batch-job).
    if target.role == "admin" and admin_repo.count_admins(db) <= 1:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Cannot delete the last admin account"},
        )

    # Снимок путей файлов до cascade-удаления записей.
    dataset_paths = admin_repo.get_dataset_storage_paths_for_user(
        db, target.id
    )
    report_paths = admin_repo.get_report_file_paths_for_user(db, target.id)

    admin_repo.delete_user(db, target)

    # После коммита чистим диск. Любая ошибка — WARNING без отката БД.
    failed_unlinks: list[str] = []
    for path in dataset_paths:
        try:
            delete_dataset_file(path)
        except OSError:
            failed_unlinks.append(path)
    for relative in report_paths:
        try:
            delete_report_file(relative)
        except OSError:
            failed_unlinks.append(relative)
    if failed_unlinks:
        logger.warning(
            "Не удалось удалить файлы при удалении пользователя %s: %s",
            user_id,
            failed_unlinks,
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
