"""
Оркестратор анализа датасета — связывает профайлер, quality_checker и БД.

Главная функция — `run_analysis`, которая запускается через FastAPI
BackgroundTasks (см. backend/app/api/analyses.py). BackgroundTask
исполняется ПОСЛЕ возврата 202 клиенту, поэтому в `run_analysis`:
1. Открывается ОТДЕЛЬНАЯ БД-сессия (не та, что у HTTP-запроса — она уже
   закрыта Depends-зависимостью).
2. Все стадии анализа выполняются в одной атомарной транзакции — либо
   все результаты сохраняются вместе, либо ничего (status=failed).

См. .knowledge/architecture/data-flow.md (шаг 4: фоновая задача анализа)
и .knowledge/troubleshooting.md (грабли с BackgroundTask и рестартом).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from app.repositories import analysis_repo
from app.repositories.quality_flag_repo import bulk_create_flags
from app.services.dataset_matcher import _load_scaler_safe, meta_features_to_embedding
from app.services.dataset_service import read_dataset_full
from app.services.profiler import compute_meta_features
from app.services.quality_checker import run_quality_checks

logger = logging.getLogger(__name__)

# Длинные стектрейсы не должны попадать в БД — это поле для UI, не для логов.
ERROR_MESSAGE_MAX_LEN = 500


def run_analysis(
    analysis_id: uuid.UUID,
    session_factory: sessionmaker,
) -> None:
    """
    Полный цикл анализа: профайлер → quality_checker → запись результатов.

    Алгоритм:
    1. Открыть свежую БД-сессию из фабрики (`SessionLocal()`).
    2. Поставить статус running, зафиксировать (commit) — чтобы фронт
       при polling видел переход pending→running без ожидания всей задачи.
    3. Прочитать датасет с диска целиком, посчитать meta-features.
    4. Применить 12 правил качества → список ORM-объектов QualityFlag.
    5. В одной транзакции:
       - bulk_create flags
       - INSERT analysis_results
       - UPDATE analyses SET status='done', finished_at=now()
       - commit
    6. При любой ошибке: rollback, status='failed',
       error_message=str(exc)[:500], finished_at=now(), commit.

    Args:
        analysis_id: UUID анализа (запись со статусом pending уже создана).
        session_factory: SessionLocal — фабрика для открытия новой сессии.
    """
    db = session_factory()
    try:
        analysis = analysis_repo.get_analysis_unscoped(db, analysis_id)  # noqa: helper ниже
        if analysis is None:
            logger.error("run_analysis: analysis %s not found", analysis_id)
            return

        # 1) pending → running, отдельный commit ради polling-видимости.
        analysis.status = "running"
        db.commit()

        # 2) Профилирование + проверка качества.
        dataset = analysis.dataset
        df = read_dataset_full(Path(dataset.storage_path), dataset.format)
        meta = compute_meta_features(df, target_col=analysis.target_column)
        flags = run_quality_checks(
            df, analysis.target_column, meta, analysis_id, db
        )

        # 3) Embedding для подбора похожих датасетов через pgvector.
        # При отсутствии scaler.pkl (модель не обучена) — анализ не валим,
        # embedding остаётся NULL: пользователь увидит результат без секции
        # «Похожие датасеты». См. .knowledge/methods/dataset-matching.md.
        embedding: list[float] | None = None
        try:
            scaler = _load_scaler_safe()
            if scaler is not None:
                embedding = meta_features_to_embedding(meta, scaler)
            else:
                logger.warning(
                    "scaler not loaded — analysis_results.embedding will be NULL "
                    "for analysis_id=%s",
                    analysis_id,
                )
        except Exception:  # noqa: BLE001 — embedding опционален
            logger.exception(
                "Failed to compute embedding for analysis_id=%s; continuing without it",
                analysis_id,
            )

        # 4) Атомарное сохранение всех результатов и финального статуса.
        bulk_create_flags(db, flags)
        analysis_repo.save_analysis_result(db, analysis_id, meta, embedding=embedding)
        analysis_repo.update_status(
            db,
            analysis_id,
            status="done",
            finished_at=datetime.now(timezone.utc),
        )
        db.commit()

    except Exception as exc:  # noqa: BLE001 — ловим всё, чтобы пометить failed
        logger.exception(
            "Analysis failed", extra={"analysis_id": str(analysis_id)}
        )
        db.rollback()
        try:
            analysis_repo.update_status(
                db,
                analysis_id,
                status="failed",
                error_message=str(exc)[:ERROR_MESSAGE_MAX_LEN],
                finished_at=datetime.now(timezone.utc),
            )
            db.commit()
        except Exception:
            logger.exception("Failed to record failure status")
            db.rollback()
    finally:
        db.close()
