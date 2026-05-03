"""
Async-оркестратор обучения baseline-моделей.

Связывает синхронный CPU-bound `baseline_trainer.train_baseline_from_df`
с FastAPI BackgroundTasks через `asyncio.to_thread`. На уровне HTTP всё это
прячется за тремя эндпоинтами в `api/analyses.py` (POST/GET /baseline).

Решение по передаче данных в поток:
- НЕ передаём в asyncio.to_thread живую SQLAlchemy-сессию: Session не
  thread-safe, и блокирующее ORM-чтение в чужом потоке плохо
  взаимодействует с пулом соединений.
- Все БД-операции (чтение df/meta/leakage_cols, обновление
  baseline_status) выполняются в основном async-потоке через sync-сессию
  (которая всё равно не блокирует event loop надолго — это лишь несколько
  быстрых SELECT/UPDATE).
- В поток уходит только чистая `train_baseline_from_df(df, meta, ...)` —
  она не трогает БД и потому безопасна.

См. .knowledge/methods/baseline-training.md, раздел «Асинхронная обёртка
в оркестраторе» и .knowledge/troubleshooting.md (грабли BackgroundTask
после рестарта — статус сбрасывается в `reset_running_to_failed`).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.quality_flag import QualityFlag
from app.models.quality_rule import QualityRule
from app.services.baseline_trainer import train_baseline_from_df
from app.services.dataset_service import read_dataset_full
from app.utils.jsonb import jsonb_safe

logger = logging.getLogger(__name__)

# Длина обрезки для baseline_error (соответствует varchar(500) в схеме).
BASELINE_ERROR_MAX_LEN = 500


def _resolve_leakage_columns(db, analysis_id: uuid.UUID) -> list[str]:
    """Список колонок с правилом LEAKAGE_SUSPICION для данного анализа."""
    stmt = (
        select(QualityFlag.context)
        .join(QualityRule, QualityFlag.rule_id == QualityRule.id)
        .where(QualityFlag.analysis_id == analysis_id)
        .where(QualityRule.code == "LEAKAGE_SUSPICION")
    )
    contexts = db.execute(stmt).scalars().all()
    cols: list[str] = []
    for ctx in contexts:
        if isinstance(ctx, dict):
            col = ctx.get("column")
            if isinstance(col, str):
                cols.append(col)
    return cols


def _set_baseline_failed(
    db,
    analysis_id: uuid.UUID,
    error_message: str,
) -> None:
    """Помечает запись baseline как failed. Используется в except-ветке."""
    try:
        db.rollback()
        ar = db.get(AnalysisResult, analysis_id)
        if ar is not None:
            ar.baseline_status = "failed"
            ar.baseline_error = error_message[:BASELINE_ERROR_MAX_LEN]
            db.commit()
    except Exception:
        logger.exception(
            "Failed to record baseline failure for analysis_id=%s", analysis_id
        )
        db.rollback()


async def run_baseline_async(
    analysis_id: uuid.UUID,
    session_factory: sessionmaker,
) -> None:
    """
    Полный цикл обучения baseline в фоне.

    Алгоритм:
    1. Открыть свежую sync-сессию (HTTP-сессия из Depends к этому моменту
       уже закрыта).
    2. Прочитать analysis + analysis_result + dataset + leakage_cols.
    3. Загрузить DataFrame с диска (это всё ещё main thread — ввод-вывод
       короткий и не нужно лочить thread pool на pandas).
    4. Перевести baseline_status в 'running' и закоммитить (для polling).
    5. Передать df/meta/leakage_cols в `asyncio.to_thread(train_baseline_from_df, ...)`
       — освобождает event loop, обучение идёт в дефолтном thread pool.
    6. После thread'а: переоткрыть запись AnalysisResult из БД (с момента
       старта могло измениться поле updated_at), сохранить
       `baseline = jsonb_safe(result)` и `baseline_status = 'done'`.
    7. При исключении: rollback, baseline_status='failed',
       baseline_error=str(exc)[:500], commit + logger.exception.

    Args:
        analysis_id: UUID анализа (PK таблицы analysis_results).
        session_factory: SessionLocal — фабрика для свежей сессии в BG-задаче.
    """
    db = session_factory()
    try:
        analysis = db.get(Analysis, analysis_id)
        ar = db.get(AnalysisResult, analysis_id)
        if analysis is None or ar is None:
            logger.error(
                "run_baseline_async: analysis or result %s not found", analysis_id
            )
            return

        target_col = analysis.target_column or ""
        task_rec: dict[str, Any] = ar.task_recommendation or {}
        # Если рекомендатер падал/был недоступен — task_recommendation = NULL,
        # и baseline корректно деградирует к NOT_READY-стабу.
        task_type = str(task_rec.get("task_type_code") or "NOT_READY")

        dataset = analysis.dataset
        df = read_dataset_full(Path(dataset.storage_path), dataset.format)
        meta = ar.meta_features or {}
        leakage_cols = _resolve_leakage_columns(db, analysis_id)

        ar.baseline_status = "running"
        ar.baseline_error = None
        db.commit()

        # Главный CPU-bound шаг — в thread pool через asyncio.to_thread.
        # train_baseline_from_df чистая функция: получает df/meta/cols,
        # ничего не пишет в БД, исключения пробрасывает наверх.
        result = await asyncio.to_thread(
            train_baseline_from_df,
            df,
            meta,
            leakage_cols,
            target_col,
            task_type,
        )

        # После to_thread сессия всё ещё та же — переоткрываем объект
        # на случай, если update_at пересчитался триггером.
        ar = db.get(AnalysisResult, analysis_id)
        if ar is None:
            logger.error(
                "run_baseline_async: analysis_result %s disappeared mid-run",
                analysis_id,
            )
            return
        ar.baseline = jsonb_safe(result)
        ar.baseline_status = "done"
        ar.baseline_error = None
        db.commit()

    except Exception as exc:  # noqa: BLE001 — ловим всё, чтобы записать failed
        logger.exception(
            "Baseline training failed for analysis_id=%s", analysis_id
        )
        _set_baseline_failed(db, analysis_id, str(exc))
    finally:
        db.close()
