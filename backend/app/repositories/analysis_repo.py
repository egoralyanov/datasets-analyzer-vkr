"""CRUD-операции над таблицами analyses и analysis_results."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.report import Report


def create_analysis(
    db: Session,
    *,
    dataset_id: uuid.UUID,
    user_id: uuid.UUID,
    target_column: str | None,
) -> Analysis:
    """Создаёт запись анализа в статусе pending — БД-фиксация через commit.

    Колонка `hinted_task_type` исторически осталась в схеме БД из Спринта 1,
    но больше не заполняется: в Спринте 3 тип задачи определяет рекомендатер,
    а не пользовательская подсказка. Старые записи остаются без изменений,
    новые записываются с NULL по умолчанию (сервер-сторона).
    """
    analysis = Analysis(
        dataset_id=dataset_id,
        user_id=user_id,
        target_column=target_column,
        status="pending",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_analysis(
    db: Session, analysis_id: uuid.UUID, user_id: uuid.UUID
) -> Analysis | None:
    """Возвращает анализ только если он принадлежит указанному пользователю (404 иначе)."""
    return db.scalar(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
    )


def get_analysis_unscoped(db: Session, analysis_id: uuid.UUID) -> Analysis | None:
    """
    Возвращает анализ без фильтра по пользователю.

    Используется только из background-задачи (analysis_service.run_analysis):
    запись была создана через API с проверкой прав, поэтому повторно
    фильтровать по user_id не нужно. Для HTTP-эндпоинтов всегда используется
    `get_analysis(..., user_id)`.
    """
    return db.get(Analysis, analysis_id)


def list_analyses(db: Session, user_id: uuid.UUID) -> list[Analysis]:
    """Все анализы пользователя в порядке от свежих к старым."""
    return list(
        db.scalars(
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .order_by(Analysis.started_at.desc())
        )
    )


def list_user_analyses_paginated(
    db: Session,
    user_id: uuid.UUID,
    *,
    page: int,
    size: int,
    status: str | None = None,
) -> tuple[list[Analysis], int]:
    """
    Пагинированный список анализов пользователя для страницы «История».

    `joinedload(Analysis.dataset)` нужен для отрисовки имени файла в строке
    списка без N+1 SELECT на каждый элемент. `joinedload(Analysis.result)`
    подтягивает task_recommendation для отображения рекомендованного типа
    задачи; result у анализа в pending/running ещё не существует, поэтому
    LEFT-JOIN, а не INNER.

    Сортировка `started_at DESC` — самые свежие сверху. created_at в модели
    отсутствует (поле ввели в analyses ещё в Спринте 1 как started_at).

    Args:
        page: номер страницы, начинается с 1.
        size: размер страницы.
        status: фильтр по статусу анализа (pending/running/done/failed)
            или None — все статусы.

    Returns:
        Кортеж `(items, total)` — срез страницы и общее число анализов
        пользователя (с применённым фильтром по status, если задан).
        Поле `pages = ceil(total / size)` вычисляется на уровне API.
    """
    base_filters = [Analysis.user_id == user_id]
    if status is not None:
        base_filters.append(Analysis.status == status)

    total = db.scalar(
        select(func.count()).select_from(Analysis).where(*base_filters)
    ) or 0

    items_stmt = (
        select(Analysis)
        .where(*base_filters)
        .options(
            joinedload(Analysis.dataset),
            joinedload(Analysis.result),
        )
        .order_by(Analysis.started_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )
    items = list(db.scalars(items_stmt).unique())
    return items, total


def update_status(
    db: Session,
    analysis_id: uuid.UUID,
    *,
    status: str,
    error_message: str | None = None,
    finished_at: datetime | None = None,
) -> None:
    """
    Меняет статус анализа.

    Не делает commit — это часть атомарной транзакции в analysis_service:
    статус, INSERT analysis_results и INSERT quality_flags должны
    зафиксироваться вместе или не зафиксироваться вовсе.
    """
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        return
    analysis.status = status
    if error_message is not None:
        analysis.error_message = error_message
    if finished_at is not None:
        analysis.finished_at = finished_at


def save_analysis_result(
    db: Session,
    analysis_id: uuid.UUID,
    meta_features: dict[str, Any],
    *,
    embedding: list[float] | None = None,
    task_recommendation: dict[str, Any] | None = None,
) -> AnalysisResult:
    """
    Создаёт запись результата анализа (или обновляет, если уже есть).

    Структура: meta_features (JSONB) — все ~30 признаков + вложенные
    distributions/correlations для UI. `embedding` — pgvector(128) для
    подбора похожих датасетов. `task_recommendation` (Phase 6) — JSONB
    с типом задачи, confidence, applied_rules и explanation. `baseline`
    заполняется отдельно в baseline_orchestrator после нажатия кнопки
    «Обучить baseline» (Phase 6).
    """
    existing = db.get(AnalysisResult, analysis_id)
    if existing is not None:
        existing.meta_features = meta_features
        if embedding is not None:
            existing.embedding = embedding
        if task_recommendation is not None:
            existing.task_recommendation = task_recommendation
        existing.updated_at = datetime.now(timezone.utc)
        return existing
    result = AnalysisResult(
        analysis_id=analysis_id,
        meta_features=meta_features,
        embedding=embedding,
        task_recommendation=task_recommendation,
    )
    db.add(result)
    return result


def reset_running_to_failed(db: Session) -> int:
    """
    Переводит «зависшие» анализы, baseline-обучения и генерации отчётов в failed.

    Вызывается при старте приложения: BackgroundTasks живут в памяти процесса,
    при перезапуске контейнера они теряются, но запись в БД остаётся
    «зависшей» в running. Сбрасываем три потока:
    1. `analyses.status='running'` → `'failed'` с error_message;
    2. `analysis_results.baseline_status='running'` → `'failed'` с baseline_error;
    3. `reports.status` ∈ {`pending`, `running`} → `'failed'` с error.
       Для отчётов сбрасываем и pending тоже: запись pending создаётся
       синхронно перед `add_task`, и если контейнер падает между INSERT'ом
       и стартом задачи — pending без задачи навсегда зависнет в очереди.

    См. .knowledge/troubleshooting.md, раздел про BackgroundTask.

    Returns:
        Суммарное число переведённых записей (analyses + baselines + reports).
    """
    running = list(db.scalars(select(Analysis).where(Analysis.status == "running")))
    for a in running:
        a.status = "failed"
        a.error_message = "Прервано перезапуском сервера"
        a.finished_at = datetime.now(timezone.utc)

    running_baselines = list(
        db.scalars(
            select(AnalysisResult).where(AnalysisResult.baseline_status == "running")
        )
    )
    for ar in running_baselines:
        ar.baseline_status = "failed"
        ar.baseline_error = "Прервано перезапуском сервера"

    running_reports = list(
        db.scalars(
            select(Report).where(Report.status.in_(("pending", "running")))
        )
    )
    for rp in running_reports:
        rp.status = "failed"
        rp.error = "Прервано перезапуском сервера"

    db.commit()
    return len(running) + len(running_baselines) + len(running_reports)
