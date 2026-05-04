"""
API-эндпоинты для PDF-отчётов.

Контракт:
- POST /api/analyses/{analysis_id}/report      → 202 + ReportCreateResponse
                                                  (или 409 + ReportConflictResponse)
- GET  /api/reports/{report_id}                → 200 + ReportRead (для polling)
- GET  /api/reports/{report_id}/download       → 200 + application/pdf

Все эндпоинты под Depends(get_current_user). Чужие отчёты и анализы дают
404 (а не 403) — чтобы не палить факт их существования.

Паттерн фон-задачи: `generate_report` — sync-функция, BackgroundTasks
автоматически запускает её через run_in_threadpool. Это отличается от
baseline_orchestrator (там async + asyncio.to_thread): в baseline есть
один изолированный CPU-bound шаг (`train_baseline_from_df`), в отчёте
CPU и БД перемешаны и общая длительность доминируется рендером —
sync-обёртка короче и не теряет в производительности.
"""
from __future__ import annotations

import logging
import urllib.parse
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.config import settings
from app.core.db import SessionLocal
from app.models.user import User
from app.repositories import analysis_repo, report_repo
from app.schemas.report import (
    ReportConflictResponse,
    ReportCreateResponse,
    ReportRead,
)
from app.services.report_service import generate_report

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


@router.post(
    "/analyses/{analysis_id}/report",
    response_model=ReportCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={409: {"model": ReportConflictResponse}},
)
def create_analysis_report(
    analysis_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Запускает фоновую генерацию PDF-отчёта по анализу.

    Гейты:
    - 404 если анализ не существует или принадлежит другому пользователю.
    - 409 + reason='analysis_not_done' если анализ ещё не завершён.
    - 409 + reason='report_in_progress' + report_id существующего, если
      отчёт уже генерируется (status pending/running) — фронт переключается
      на polling существующего.

    При успехе создаётся Report в pending, BackgroundTask ставит
    `generate_report` в очередь, возвращается 202 + ReportCreateResponse.
    """
    analysis = analysis_repo.get_analysis(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Анализ не найден"
        )
    if analysis.status != "done":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ReportConflictResponse(
                detail="Сначала дождитесь завершения анализа",
                reason="analysis_not_done",
            ).model_dump(mode="json"),
        )

    existing = report_repo.get_active_report_for_analysis(db, analysis_id)
    if existing is not None:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=ReportConflictResponse(
                detail="Отчёт уже генерируется",
                reason="report_in_progress",
                report_id=existing.id,
                status=existing.status,
            ).model_dump(mode="json"),
        )

    report = report_repo.create_report(
        db, analysis_id=analysis.id, user_id=current_user.id
    )
    # SessionLocal — фабрика. BackgroundTask откроет свежую сессию внутри
    # generate_report (HTTP-сессия из Depends к моменту запуска фон-задачи
    # уже закрыта).
    background_tasks.add_task(generate_report, report.id, SessionLocal)
    return ReportCreateResponse.model_validate(report)


@router.get("/reports/{report_id}", response_model=ReportRead)
def get_report_status(
    report_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportRead:
    """Текущее состояние отчёта (для polling из ReportDownloadButton)."""
    report = report_repo.get_report(db, report_id, current_user.id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Отчёт не найден"
        )
    return ReportRead.model_validate(report)


@router.get("/reports/{report_id}/download")
def download_report(
    report_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Отдаёт сгенерированный PDF.

    404 если: отчёт не найден, не успешен, или файл отсутствует на диске
    (например, повреждён volume — пишем warning в лог). Status коды все
    одинаковые — не палим, чем именно отчёт не подошёл.

    Имя файла формируется в RFC 5987 (`filename` ASCII-fallback +
    `filename*=UTF-8''...` для современных браузеров) — кириллица в
    `dataset.original_filename` отображается корректно, без квадратиков
    и без обрезания на запятых.
    """
    report = report_repo.get_report(db, report_id, current_user.id)
    if report is None or report.status != "success" or not report.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Отчёт не найден"
        )

    pdf_path = Path(settings.REPORTS_DIR) / report.file_path
    if not pdf_path.exists():
        logger.warning(
            "Report %s marked success but file missing on disk: %s",
            report.id, pdf_path,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Отчёт не найден"
        )

    analysis = analysis_repo.get_analysis(db, report.analysis_id, current_user.id)
    dataset_stem = "report"
    if analysis is not None and analysis.dataset is not None:
        dataset_stem = Path(analysis.dataset.original_filename).stem or "report"
    date_str = report.created_at.strftime("%Y-%m-%d")
    pretty_name = f"report_{dataset_stem}_{date_str}.pdf"
    encoded_name = urllib.parse.quote(pretty_name, safe="")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="report.pdf"; '
                f"filename*=UTF-8''{encoded_name}"
            )
        },
    )
