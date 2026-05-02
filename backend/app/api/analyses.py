"""
API-эндпоинты для запуска и получения анализа.

Контракт:
- POST /api/datasets/{dataset_id}/analyze   → 202 + AnalysisResponse
- GET  /api/analyses                        → list[AnalysisResponse]
- GET  /api/analyses/{analysis_id}          → AnalysisResponse (для polling)
- GET  /api/analyses/{analysis_id}/result   → AnalysisResultResponse
                                              (409 если status != done)

Все эндпоинты под Depends(get_current_user). Чужие анализы и датасеты
дают 404 (не 403) — чтобы не палить факт их существования.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.db import SessionLocal
from app.models.user import User
from app.repositories import analysis_repo, dataset_repo
from app.repositories.quality_flag_repo import get_flags_for_analysis
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisResultResponse,
    QualityFlagResponse,
    StartAnalysisRequest,
)
from app.services.analysis_service import run_analysis

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
        hinted_task_type=payload.hinted_task_type,
    )

    # SessionLocal — фабрика, не активная сессия. BackgroundTask откроет
    # свежую сессию внутри run_analysis (см. analysis_service).
    background_tasks.add_task(run_analysis, analysis.id, SessionLocal)
    return AnalysisResponse.model_validate(analysis)


@router.get("/analyses", response_model=list[AnalysisResponse])
def list_my_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AnalysisResponse]:
    """Все свои анализы в порядке от свежих к старым."""
    items = analysis_repo.list_analyses(db, current_user.id)
    return [AnalysisResponse.model_validate(a) for a in items]


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
    """Полный результат — meta-features + флаги. Только для done-анализов."""
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
    return AnalysisResultResponse(
        analysis=AnalysisResponse.model_validate(analysis),
        meta_features=analysis.result.meta_features,
        flags=flags,
    )
