"""CRUD-операции над таблицей users + per-user агрегаты для дашборда."""
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User


def create_user(db: Session, *, email: str, username: str, password_hash: str) -> User:
    user = User(email=email, username=username, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email))


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def get_user_by_id(db: Session, user_id: uuid.UUID) -> User | None:
    return db.get(User, user_id)


def update_user(db: Session, user: User, fields: dict[str, Any]) -> User:
    for key, value in fields.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user


def compute_user_stats(db: Session, user_id: uuid.UUID) -> dict[str, int]:
    """
    Личные счётчики пользователя для GET /api/me/stats (Спринт 6, Phase 7):
    датасеты, анализы, успешные анализы, PDF-отчёты.

    «Успешный анализ» = `Analysis.status == 'done'` (без проверки
    task_recommendation). То же определение использует admin_repo для
    `analyses_success_rate` в /admin/stats — расхождение между админ- и
    user-метриками одного и того же пользователя было бы багом ожидания.
    NOT_READY — корректный вердикт «данных недостаточно», а не неудача.

    Считаем `reports_count` только по `status='success'` — другие
    статусы (`pending`, `failed`) для UI не интересны, как и в
    admin/stats и /datasets/{id}/usage.
    """
    datasets_count = db.scalar(
        select(func.count())
        .select_from(Dataset)
        .where(Dataset.user_id == user_id)
    ) or 0
    analyses_count = db.scalar(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.user_id == user_id)
    ) or 0
    successful_analyses_count = db.scalar(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.user_id == user_id, Analysis.status == "done")
    ) or 0
    # Report имеет собственный user_id (составной индекс
    # ix_reports_user_status), JOIN с analyses не нужен.
    reports_count = db.scalar(
        select(func.count())
        .select_from(Report)
        .where(Report.user_id == user_id, Report.status == "success")
    ) or 0

    return {
        "datasets_count": int(datasets_count),
        "analyses_count": int(analyses_count),
        "successful_analyses_count": int(successful_analyses_count),
        "reports_count": int(reports_count),
    }
