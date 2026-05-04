"""
Запросы для админ-панели: агрегированная статистика и пагинированный
список пользователей с числом датасетов/анализов.

Использует scalar-subqueries вместо LEFT JOIN ... GROUP BY — это избавляет
от GROUP BY на всех колонках User и оставляет основной запрос линейным
по структуре (см. .knowledge/architecture/database.md, раздел 1).
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User


def compute_admin_stats(db: Session) -> dict[str, int | float | None]:
    """
    Шесть метрик для дашборда админ-панели:
    - total_users / total_datasets / total_analyses / total_reports — счётчики;
    - analyses_success_rate = done / total (None если total=0);
    - reports_success_rate  = success / total (None если total=0).

    Шесть отдельных простых COUNT'ов вместо одного «умного» UNION/CTE —
    на 5-100 записях разница в latency миллисекунды, читаемость важнее.
    """
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    total_datasets = db.scalar(select(func.count()).select_from(Dataset)) or 0
    total_analyses = db.scalar(select(func.count()).select_from(Analysis)) or 0
    total_reports = db.scalar(select(func.count()).select_from(Report)) or 0

    done_analyses = db.scalar(
        select(func.count()).select_from(Analysis).where(Analysis.status == "done")
    ) or 0
    success_reports = db.scalar(
        select(func.count()).select_from(Report).where(Report.status == "success")
    ) or 0

    analyses_rate: float | None = (
        done_analyses / total_analyses if total_analyses else None
    )
    reports_rate: float | None = (
        success_reports / total_reports if total_reports else None
    )

    return {
        "total_users": total_users,
        "total_datasets": total_datasets,
        "total_analyses": total_analyses,
        "total_reports": total_reports,
        "analyses_success_rate": analyses_rate,
        "reports_success_rate": reports_rate,
    }


def list_users_paginated(
    db: Session, *, page: int, size: int
) -> tuple[list[tuple[User, int, int]], int]:
    """
    Возвращает срез страницы пользователей вместе с числом их датасетов
    и анализов (для строки таблицы админки) + общее число пользователей.

    Сортировка `created_at DESC` — новые регистрации сверху, типичный
    паттерн админки. Для агрегатов используется correlate-subquery
    (`scalar_subquery()` под `User.id`), чтобы не делать GROUP BY на
    всех колонках User.
    """
    datasets_count_sq = (
        select(func.count(Dataset.id))
        .where(Dataset.user_id == User.id)
        .correlate(User)
        .scalar_subquery()
    )
    analyses_count_sq = (
        select(func.count(Analysis.id))
        .where(Analysis.user_id == User.id)
        .correlate(User)
        .scalar_subquery()
    )

    total = db.scalar(select(func.count()).select_from(User)) or 0

    stmt = (
        select(
            User,
            datasets_count_sq.label("datasets_count"),
            analyses_count_sq.label("analyses_count"),
        )
        .order_by(User.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )
    rows = db.execute(stmt).all()
    items: list[tuple[User, int, int]] = [
        (row.User, int(row.datasets_count), int(row.analyses_count))
        for row in rows
    ]
    return items, total
