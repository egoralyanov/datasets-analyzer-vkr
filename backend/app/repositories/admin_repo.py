"""
Запросы для админ-панели: агрегированная статистика и пагинированный
список пользователей с числом датасетов/анализов.

Использует scalar-subqueries вместо LEFT JOIN ... GROUP BY — это избавляет
от GROUP BY на всех колонках User и оставляет основной запрос линейным
по структуре (см. .project_docs/architecture/database.md, раздел 1).
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


def count_admins(db: Session) -> int:
    """
    Число пользователей с ролью admin.

    Используется в `DELETE /api/admin/users/{id}` (Спринт 6, Phase 4.4)
    как defense-in-depth: запрещаем удаление, если удаляемый — admin и
    в системе он единственный admin. На практике этот кейс закрыт раньше
    self-check'ом (для отправки DELETE нужен сам admin), но защита
    полезна на случай будущих CLI-сценариев или второго пути удаления.
    """
    return int(
        db.scalar(
            select(func.count())
            .select_from(User)
            .where(User.role == "admin")
        )
        or 0
    )


def get_dataset_storage_paths_for_user(
    db: Session, user_id
) -> list[str]:
    """
    Абсолютные пути файлов всех датасетов пользователя.

    Снимок собирается ДО `db.delete(user)` — после cascade-удаления
    записей `datasets` пути было бы негде взять. Используется в
    DELETE /api/admin/users/{id} для зачистки storage с диска
    (Спринт 6, Phase 4.4).
    """
    rows = db.execute(
        select(Dataset.storage_path).where(Dataset.user_id == user_id)
    ).all()
    return [row[0] for row in rows if row[0]]


def get_report_file_paths_for_user(
    db: Session, user_id
) -> list[str]:
    """
    Относительные пути PDF-файлов всех success-отчётов пользователя.

    `Report.file_path` — относительно `settings.REPORTS_DIR`. Фильтр
    `status='success' AND file_path IS NOT NULL` по той же причине,
    что в `analysis_repo.get_report_file_paths_for_analysis`: у
    failed/pending файла на диске нет.
    """
    rows = db.execute(
        select(Report.file_path).where(
            Report.user_id == user_id,
            Report.status == "success",
            Report.file_path.is_not(None),
        )
    ).all()
    return [row[0] for row in rows if row[0]]


def delete_user(db: Session, user: User) -> None:
    """
    Удаляет пользователя; FK ondelete=CASCADE по цепочке унесёт
    datasets → analyses → analysis_results / quality_flags / reports.
    Файлы на диске удаляет эндпоинт после коммита.
    """
    db.delete(user)
    db.commit()


def compute_user_aggregates(db: Session, user_id) -> tuple[int, int, int]:
    """
    Возвращает `(datasets_count, analyses_count, reports_count)` для одного
    пользователя. Используется в GET /api/admin/users/{id} (Спринт 6,
    Phase 4.5) — детальная карточка в модалке админ-панели.

    `reports_count` фильтрует по `status='success'` — та же семантика,
    что в /api/datasets/{id}/usage. Связь Report→User денормализована
    (есть Report.user_id), JOIN с analyses не требуется.
    """
    datasets_count = int(
        db.scalar(
            select(func.count())
            .select_from(Dataset)
            .where(Dataset.user_id == user_id)
        )
        or 0
    )
    analyses_count = int(
        db.scalar(
            select(func.count())
            .select_from(Analysis)
            .where(Analysis.user_id == user_id)
        )
        or 0
    )
    reports_count = int(
        db.scalar(
            select(func.count())
            .select_from(Report)
            .where(
                Report.user_id == user_id,
                Report.status == "success",
            )
        )
        or 0
    )
    return datasets_count, analyses_count, reports_count


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
