"""Health-check эндпоинт. Используется healthcheck'ом docker-compose и фронтом."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Возвращает статус сервиса."""
    return {"status": "ok", "service": "backend"}
