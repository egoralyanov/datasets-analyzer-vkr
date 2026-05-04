"""
API-эндпоинты для запуска и получения анализа.

Контракт:
- POST /api/datasets/{dataset_id}/analyze       → 202 + AnalysisResponse
- GET  /api/analyses                            → list[AnalysisResponse]
- GET  /api/analyses/{analysis_id}              → AnalysisResponse (для polling)
- GET  /api/analyses/{analysis_id}/result       → AnalysisResultResponse
                                                  (409 если status != done)
- POST /api/analyses/{analysis_id}/baseline     → 202 (новый запуск) /
                                                  200 (уже done, идемпотентность) /
                                                  409 (уже running либо анализ не done)
- GET  /api/analyses/{analysis_id}/baseline     → BaselineResultResponse
                                                  (404 если baseline_status='not_started')
- GET  /api/analyses/{analysis_id}/similar      → list[SimilarDatasetResponse]
                                                  (пустой список если embedding=NULL)

Все эндпоинты под Depends(get_current_user). Чужие анализы и датасеты
дают 404 (не 403) — чтобы не палить факт их существования.
"""
from __future__ import annotations

import uuid

from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.db import SessionLocal
from app.models.analysis_result import AnalysisResult
from app.models.user import User
from app.repositories import analysis_repo, dataset_repo
from app.repositories.quality_flag_repo import get_flags_for_analysis
from app.schemas.analysis import (
    AnalysisListItem,
    AnalysisListResponse,
    AnalysisResponse,
    AnalysisResultResponse,
    QualityFlagResponse,
    StartAnalysisRequest,
)
from app.schemas.baseline import (
    BaselineResultResponse,
    BaselineStartResponse,
    SimilarDatasetResponse,
)
from app.services.analysis_service import run_analysis
from app.services.baseline_orchestrator import run_baseline_async
from app.services.dataset_matcher import find_similar_datasets

router = APIRouter(tags=["analyses"])


@router.post(
    "/datasets/{dataset_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_analysis(
    dataset_id: uuid.UUID,
    payload: StartAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """
    Запускает фоновый анализ датасета.

    Проверяет принадлежность датасета текущему пользователю, создаёт запись
    `analyses` со статусом pending, ставит BackgroundTask на полное
    выполнение анализа и сразу возвращает 202 Accepted с ID анализа —
    дальше фронт делает polling через GET /api/analyses/{id}.
    """
    dataset = dataset_repo.get_dataset(db, dataset_id, current_user.id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Датасет не найден"
        )

    # Если указан target_column, валидируем что он есть среди колонок.
    # Делать read_dataset_full ради валидации дорого (для больших файлов
    # это секунды) — поэтому проверяем по preview один раз. NB: это «лёгкая»
    # валидация; полное чтение потом в background-задаче всё равно случится.
    if payload.target_column is not None:
        from pathlib import Path

        from app.services.dataset_service import read_dataset_preview

        try:
            preview = read_dataset_preview(Path(dataset.storage_path), dataset.format, max_rows=1)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Не удалось прочитать датасет: {exc}",
            )
        if payload.target_column not in preview["columns"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Колонка «{payload.target_column}» отсутствует в датасете",
            )

    analysis = analysis_repo.create_analysis(
        db,
        dataset_id=dataset.id,
        user_id=current_user.id,
        target_column=payload.target_column,
    )

    # SessionLocal — фабрика, не активная сессия. BackgroundTask откроет
    # свежую сессию внутри run_analysis (см. analysis_service).
    background_tasks.add_task(run_analysis, analysis.id, SessionLocal)
    return AnalysisResponse.model_validate(analysis)


@router.get("/analyses", response_model=AnalysisListResponse)
def list_my_analyses(
    page: int = Query(1, ge=1, description="Номер страницы (с 1)"),
    size: int = Query(20, ge=1, le=100, description="Размер страницы (1..100)"),
    status_filter: Literal["pending", "running", "done", "failed"] | None = Query(
        None,
        alias="status",
        description="Опциональный фильтр по статусу",
    ),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisListResponse:
    """
    Пагинированный список анализов пользователя.

    Сортировка started_at DESC, joinedload подтягивает Dataset и
    AnalysisResult в одном запросе — без N+1. Контракт ответа:
    {items, total, page, size, pages} — стандартный для пагинированных
    списков.
    """
    items, total = analysis_repo.list_user_analyses_paginated(
        db, current_user.id, page=page, size=size, status=status_filter,
    )
    pages = (total + size - 1) // size if total else 0
    return AnalysisListResponse(
        items=[
            AnalysisListItem(
                id=a.id,
                dataset_id=a.dataset_id,
                dataset_name=a.dataset.original_filename if a.dataset else "—",
                status=a.status,
                target_column=a.target_column,
                recommended_task_type=(
                    a.result.task_recommendation.get("task_type_code")
                    if a.result and a.result.task_recommendation
                    else None
                ),
                started_at=a.started_at,
                finished_at=a.finished_at,
            )
            for a in items
        ],
        total=total,
        page=page,
        size=size,
        pages=pages,
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_status(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    """Лёгкая модель — для polling статуса фронтом."""
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )
    return AnalysisResponse.model_validate(analysis)


@router.get(
    "/analyses/{analysis_id}/result", response_model=AnalysisResultResponse
)
def get_analysis_result(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResultResponse:
    """Полный результат — meta-features + флаги + рекомендация типа задачи + embedding.

    Только для done-анализов (409 иначе). task_recommendation и embedding
    могут быть None — например, если scaler/модель отсутствуют (graceful degradation).
    """
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )
    if analysis.status != "done":
        # 409 Conflict — анализ есть, но запросить полный результат пока нельзя
        # (фронт сам сделает polling и попадёт сюда после перехода в done).
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Анализ ещё не завершён (статус: {analysis.status})",
        )

    if analysis.result is None:
        # Не должно случаться: при status=done результат обязан быть.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Результат отсутствует, хотя анализ помечен как done",
        )

    flag_rows = get_flags_for_analysis(db, analysis_id)
    flags = [
        QualityFlagResponse(
            rule_code=rule.code,
            severity=rule.severity,
            rule_name=rule.name,
            message=flag.message,
            context=flag.context,
        )
        for flag, rule in flag_rows
    ]
    # pgvector возвращает embedding как numpy.ndarray — приводим к list[float]
    # для надёжной JSON-сериализации.
    embedding_value = analysis.result.embedding
    embedding_list: list[float] | None = (
        [float(v) for v in embedding_value] if embedding_value is not None else None
    )
    return AnalysisResultResponse(
        analysis=AnalysisResponse.model_validate(analysis),
        meta_features=analysis.result.meta_features,
        flags=flags,
        task_recommendation=analysis.result.task_recommendation,
        embedding=embedding_list,
    )


# =============================================================================
#                               BASELINE
# =============================================================================


@router.post("/analyses/{analysis_id}/baseline", response_model=BaselineStartResponse)
def start_baseline(
    analysis_id: uuid.UUID,
    response: Response,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BaselineStartResponse:
    """Запускает обучение baseline-моделей в фоне.

    Идемпотентность:
    - повторный POST в `running` → 409 «Baseline уже обучается»;
    - повторный POST после `done` → 200 OK + текущее состояние (без перезапуска);
    - POST на анализ со статусом != done → 409.

    Перезапуск с `failed` или `not_started` стартует свежую BG-задачу и
    возвращает 202.
    """
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )
    if analysis.status != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Сначала дождитесь завершения анализа",
        )

    ar = db.get(AnalysisResult, analysis_id)
    if ar is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Результат анализа отсутствует, хотя статус done",
        )

    if ar.baseline_status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Baseline уже обучается"
        )
    if ar.baseline_status == "done":
        # Идемпотентность: возвращаем 200 OK с текущим done-статусом,
        # но не перезапускаем (фронт может тут же запросить GET /baseline).
        response.status_code = status.HTTP_200_OK
        return BaselineStartResponse(analysis_id=analysis_id, baseline_status="done")

    background_tasks.add_task(run_baseline_async, analysis_id, SessionLocal)
    response.status_code = status.HTTP_202_ACCEPTED
    return BaselineStartResponse(analysis_id=analysis_id, baseline_status="running")


@router.get(
    "/analyses/{analysis_id}/baseline", response_model=BaselineResultResponse
)
def get_baseline(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BaselineResultResponse:
    """Текущее состояние baseline-обучения для анализа.

    404 если ни разу не запускали (`baseline_status='not_started'`) — для UI
    это сигнал показать кнопку «Обучить baseline». Все остальные статусы
    (running / done / failed) отдаются как 200 + структура с baseline и error.
    """
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )

    ar = db.get(AnalysisResult, analysis_id)
    if ar is None or ar.baseline_status == "not_started":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Baseline не запускался"
        )

    return BaselineResultResponse(
        baseline_status=ar.baseline_status,
        baseline=ar.baseline,
        baseline_error=ar.baseline_error,
    )


@router.get(
    "/analyses/{analysis_id}/similar",
    response_model=list[SimilarDatasetResponse],
)
def get_similar(
    analysis_id: uuid.UUID,
    top_k: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SimilarDatasetResponse]:
    """Top-K похожих датасетов из каталога через pgvector.

    Если у анализа нет embedding (scaler не был доступен в момент анализа) —
    возвращаем пустой список без ошибки: фронт покажет пустую секцию.

    Фильтр task_type_code берётся из task_recommendation, если он определён —
    это даёт более релевантные «похожие» из той же подкатегории. Если
    рекомендатер падал, фильтр None и матчер ищет среди всех типов.
    """
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )

    ar = db.get(AnalysisResult, analysis_id)
    if ar is None or ar.embedding is None:
        return []

    task_type_filter: str | None = None
    if ar.task_recommendation:
        candidate = ar.task_recommendation.get("task_type_code")
        if isinstance(candidate, str):
            task_type_filter = candidate

    # pgvector хранит numpy.ndarray; репозиторий ожидает list[float].
    query_embedding = [float(v) for v in ar.embedding]

    similar = find_similar_datasets(
        db,
        query_embedding,
        task_type_filter=task_type_filter,
        top_k=top_k,
        metric="cosine",
    )
    # SimilarDatasetResponse читает атрибут `distance`, прицепленный
    # репозиторием через setattr — model_validate (ConfigDict.from_attributes=True)
    # подхватит его наравне с обычными ORM-полями.
    return [SimilarDatasetResponse.model_validate(s) for s in similar]
