"""
Оркестратор анализа датасета — связывает профайлер, quality_checker,
рекомендатер типа задачи, embedding для поиска похожих и БД.

Главная функция — `run_analysis`, которая запускается через FastAPI
BackgroundTasks (см. backend/app/api/analyses.py). BackgroundTask
исполняется ПОСЛЕ возврата 202 клиенту, поэтому в `run_analysis`:
1. Открывается ОТДЕЛЬНАЯ БД-сессия (не та, что у HTTP-запроса — она уже
   закрыта Depends-зависимостью).
2. Все стадии анализа выполняются в одной атомарной транзакции — либо
   все результаты сохраняются вместе, либо ничего (status=failed).

Стадии (после Спринта 3):
1. profiler.compute_meta_features → meta-признаки.
2. quality_checker.run_quality_checks → флаги качества.
3. task_recommender.recommend_task → рекомендация типа ML-задачи (JSONB).
4. dataset_matcher.meta_features_to_embedding → 128-вектор для pgvector.
5. Запись всего в analysis_results + flags в одной транзакции.

Ошибки рекомендатера и embedding'а **не валят** анализ — они опциональны
для основного отчёта. Если task_recommender падает, в analysis_results
остаётся task_recommendation=NULL и UI показывает «Не удалось определить
тип задачи». Аналогично с embedding (нет похожих датасетов).

См. .knowledge/architecture/data-flow.md (шаг 4: фоновая задача анализа)
и .knowledge/troubleshooting.md (грабли с BackgroundTask и рестартом).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models.quality_rule import QualityRule
from app.repositories import analysis_repo
from app.repositories.quality_flag_repo import bulk_create_flags
from app.services.dataset_matcher import _load_scaler_safe, meta_features_to_embedding
from app.services.dataset_service import read_dataset_full
from app.services.profiler import compute_meta_features
from app.services.quality_checker import run_quality_checks
from app.services.task_recommender import recommend_task
from app.utils.jsonb import jsonb_safe

logger = logging.getLogger(__name__)

# Длинные стектрейсы не должны попадать в БД — это поле для UI, не для логов.
ERROR_MESSAGE_MAX_LEN = 500


def run_analysis(
    analysis_id: uuid.UUID,
    session_factory: sessionmaker,
) -> None:
    """
    Полный цикл анализа: профайлер → quality_checker → recommend_task →
    embedding → запись результатов.

    Алгоритм:
    1. Открыть свежую БД-сессию из фабрики (`SessionLocal()`).
    2. Поставить статус running, зафиксировать (commit) — чтобы фронт
       при polling видел переход pending→running без ожидания всей задачи.
    3. Прочитать датасет с диска целиком, посчитать meta-features.
    4. Применить 12 правил качества → список ORM-объектов QualityFlag.
    5. Рекомендовать тип задачи (recommend_task) — best-effort, ошибка
       не валит анализ.
    6. Посчитать embedding (best-effort, как и в Спринте 3 Phase 4).
    7. Санитизировать meta и task_recommendation через jsonb_safe —
       NaN/Infinity ломают JSONB (грабля Phase 4 — корреляции для
       константных колонок дают NaN).
    8. В одной транзакции:
       - bulk_create flags
       - INSERT analysis_results (с task_recommendation и embedding)
       - UPDATE analyses SET status='done', finished_at=now()
       - commit
    9. При любой ошибке: rollback, status='failed',
       error_message=str(exc)[:500], finished_at=now(), commit.

    Args:
        analysis_id: UUID анализа (запись со статусом pending уже создана).
        session_factory: SessionLocal — фабрика для открытия новой сессии.
    """
    db = session_factory()
    try:
        analysis = analysis_repo.get_analysis_unscoped(db, analysis_id)
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

        # 3) Рекомендация типа задачи. Чтобы передать в recommend_task
        # активные quality-правила, резолвим rule_id флагов в их code:
        # это нужно для критических флагов из Ветки 5 recommender-rules.md
        # (TARGET_MISSING / LEAKAGE_SUSPICION / SMALL_DATASET → предупреждения
        # в applied_rules рекомендации).
        active_quality_codes: list[str] = []
        flag_rule_ids = {f.rule_id for f in flags}
        if flag_rule_ids:
            try:
                active_quality_codes = list(
                    db.execute(
                        select(QualityRule.code).where(
                            QualityRule.id.in_(flag_rule_ids)
                        )
                    ).scalars()
                )
            except Exception:  # noqa: BLE001 — ошибка резолвинга не валит анализ
                logger.exception(
                    "Failed to resolve quality_rule codes for analysis_id=%s",
                    analysis_id,
                )

        task_rec_dict: dict | None = None
        try:
            task_rec = recommend_task(meta, analysis.target_column, active_quality_codes)
            task_rec_dict = task_rec.model_dump()
        except Exception:  # noqa: BLE001 — рекомендатер опционален
            logger.exception(
                "task_recommender failed for analysis_id=%s; continuing without it",
                analysis_id,
            )

        # 4) Embedding для подбора похожих датасетов через pgvector.
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

        # 5) Санитизация перед записью в JSONB. NaN/Infinity встречаются:
        # - в correlation_matrix для константных колонок (профайлер);
        # - в ml_probabilities при вырожденных классах (рекомендатер).
        # Без этого PostgreSQL отвергает INSERT с
        # `Token "NaN" is invalid`.
        meta_safe = jsonb_safe(meta)
        task_rec_safe = jsonb_safe(task_rec_dict) if task_rec_dict is not None else None

        # 6) Атомарное сохранение всех результатов и финального статуса.
        bulk_create_flags(db, flags)
        analysis_repo.save_analysis_result(
            db,
            analysis_id,
            meta_safe,
            embedding=embedding,
            task_recommendation=task_rec_safe,
        )
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
