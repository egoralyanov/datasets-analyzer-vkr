"""
Личные эндпоинты текущего пользователя (Sprint 6, Phase 7).

Контракт:
- GET /api/me/stats → 200 + UserStats {datasets_count, analyses_count,
  successful_analyses_count, reports_count}.
- 401 без auth — стандартное поведение Depends(get_current_user).

Помещён в отдельный router /me, потому что /auth уже занят сценариями
аутентификации (login/register/logout/me-profile/password). Личная
аналитика юзера — про него самого, но это не auth-flow; держим в
самостоятельном модуле, чтобы маршруты читались по логике.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.repositories import user_repo
from app.schemas.user import UserStats

router = APIRouter(prefix="/me", tags=["me"])


@router.get("/stats", response_model=UserStats)
def get_my_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserStats:
    """Личные счётчики: датасеты, анализы, успешные анализы, PDF-отчёты."""
    return UserStats(**user_repo.compute_user_stats(db, current_user.id))
