"""Точка входа FastAPI-приложения."""
from fastapi import FastAPI

from app.api import health
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Все роуты публикуются под префиксом /api — фронт ходит сюда через nginx-прокси.
app.include_router(health.router, prefix="/api")
