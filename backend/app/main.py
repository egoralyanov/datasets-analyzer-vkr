"""Точка входа FastAPI-приложения."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import analyses, auth, datasets, health
from app.config import settings
from app.core.db import SessionLocal
from app.repositories.analysis_repo import reset_running_to_failed

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan-контекст: выполняется один раз при старте и один раз при остановке.

    На старте: переводит «зависшие» running-анализы в failed.
    BackgroundTasks живут в памяти процесса — при перезапуске контейнера они
    теряются, но запись в БД остаётся. См. .knowledge/troubleshooting.md.
    """
    db = SessionLocal()
    try:
        recovered = reset_running_to_failed(db)
        if recovered:
            logger.warning(
                "Reset %s stuck running analyses to failed at startup", recovered
            )
    finally:
        db.close()
    yield
    # Здесь можно было бы делать graceful shutdown — но текущая
    # архитектура не требует освобождения ресурсов.


app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Все роуты публикуются под префиксом /api — фронт ходит сюда через nginx-прокси.
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(analyses.router, prefix="/api")
