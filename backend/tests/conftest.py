"""
Общие фикстуры pytest.

Тесты используют ту же БД `analyzer` (production-like: PostgreSQL + pgvector).
Изоляция между тестами — через TRUNCATE users CASCADE в teardown фикстуры
db_session: каскад чистит datasets и analyses. Файлы датасетов на диске
чистит фикстура datasets_storage_cleanup (autouse).
"""
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.db import SessionLocal, engine
from app.core.security import create_access_token, hash_password
from app.main import app
from app.models.user import User


@pytest.fixture
def client() -> TestClient:
    """HTTP-клиент к FastAPI-приложению (in-process, без поднятия сервера)."""
    return TestClient(app)


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Сессия БД с TRUNCATE-очисткой после теста (CASCADE на datasets и analyses)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE users CASCADE"))


@pytest.fixture
def test_user(db_session: Session) -> Callable[..., dict[str, Any]]:
    """
    Фабрика тестовых пользователей.

    Возвращает dict с user-моделью и plain-паролем (нужен для последующего логина).
    Использует настоящий hash_password() — без моков.
    """

    def _make(
        email: str | None = None,
        username: str | None = None,
        password: str = "Strong123!",
        role: str = "user",
    ) -> dict[str, Any]:
        suffix = uuid.uuid4().hex[:8]
        user = User(
            email=email or f"test-{suffix}@example.com",
            username=username or f"user-{suffix}",
            password_hash=hash_password(password),
            role=role,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return {"user": user, "password": password}

    return _make


@pytest.fixture
def auth_headers() -> Callable[[User], dict[str, str]]:
    """Фабрика заголовков Authorization: Bearer <token> для произвольного пользователя."""

    def _make(user: User) -> dict[str, str]:
        token = create_access_token(user.id, user.role)
        return {"Authorization": f"Bearer {token}"}

    return _make


@pytest.fixture(autouse=True)
def datasets_storage_cleanup() -> Iterator[None]:
    """Чистит файловое хранилище датасетов после каждого теста."""
    yield
    storage = Path(settings.DATASETS_DIR)
    if storage.exists():
        for child in storage.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
