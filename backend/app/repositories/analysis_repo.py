"""CRUD-операции над таблицами analyses и analysis_results."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult


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
    Переводит «зависшие» анализы и baseline-обучения в статус failed.

    Вызывается при старте приложения: BackgroundTasks живут в памяти процесса,
    при перезапуске контейнера они теряются, но запись в БД остаётся
    «зависшей» в running. Сбрасываем сразу два статуса:
    1. `analyses.status='running'` → `'failed'` с error_message;
    2. `analysis_results.baseline_status='running'` → `'failed'` с baseline_error.

    См. .knowledge/troubleshooting.md, раздел про BackgroundTask.

    Returns:
        Суммарное число переведённых записей (analyses + baselines).
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

    db.commit()
    return len(running) + len(running_baselines)
