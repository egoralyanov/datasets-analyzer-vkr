"""
Интеграционные тесты GET /api/admin/users/{id} (Спринт 6, Phase 4.5).

Контракт:
- 200 + AdminUserDetail (id, email, username, role, created_at,
  datasets_count, analyses_count, reports_count) — для admin.
- 401 — без auth.
- 403 — auth есть, но не admin.
- 404 — несуществующий user_id.

`reports_count` фильтрует по status='success' (та же семантика, что в
GET /api/datasets/{id}/usage).
"""
from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User


def _make_dataset(db: Session, user: User) -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        original_filename="iris.csv",
        storage_path=f"/data/datasets/{user.id}/iris.csv",
        file_size_bytes=4096,
        file_hash=secrets.token_hex(32),
        format="csv",
        n_rows=150,
        n_cols=5,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def _make_analysis(
    db: Session, user: User, dataset: Dataset, *, status: str = "done"
) -> Analysis:
    analysis = Analysis(
        dataset_id=dataset.id,
        user_id=user.id,
        target_column=None,
        status=status,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _make_report(
    db: Session, user: User, analysis: Analysis, *, status: str
) -> Report:
    report = Report(
        analysis_id=analysis.id,
        user_id=user.id,
        status=status,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def test_admin_gets_user_details(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Полный сценарий: 1 датасет, 2 анализа, 1 success + 1 failed report → reports_count=1."""
    admin = test_user(username="adm", role="admin")["user"]
    target = test_user(
        username="target",
        email="target@example.com",
    )["user"]

    dataset = _make_dataset(db_session, target)
    a1 = _make_analysis(db_session, target, dataset)
    a2 = _make_analysis(db_session, target, dataset)
    _make_report(db_session, target, a1, status="success")
    _make_report(db_session, target, a2, status="failed")

    response = client.get(
        f"/api/admin/users/{target.id}", headers=auth_headers(admin)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(target.id)
    assert body["email"] == "target@example.com"
    assert body["username"] == "target"
    assert body["role"] == "user"
    assert "created_at" in body
    assert body["datasets_count"] == 1
    assert body["analyses_count"] == 2
    # failed-report не учитывается.
    assert body["reports_count"] == 1


def test_admin_gets_user_with_zero_datasets(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Smoke на агрегаты для пустого пользователя — все три счётчика 0."""
    admin = test_user(username="adm", role="admin")["user"]
    target = test_user(username="lonely")["user"]

    response = client.get(
        f"/api/admin/users/{target.id}", headers=auth_headers(admin)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["datasets_count"] == 0
    assert body["analyses_count"] == 0
    assert body["reports_count"] == 0


def test_user_gets_403(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Не-admin → 403 (это /api/admin/*)."""
    plain = test_user(username="plain")["user"]
    target = test_user(username="other")["user"]

    response = client.get(
        f"/api/admin/users/{target.id}", headers=auth_headers(plain)
    )
    assert response.status_code == 403


def test_unauthenticated_gets_401(client: TestClient) -> None:
    response = client.get(f"/api/admin/users/{uuid.uuid4()}")
    assert response.status_code == 401


def test_nonexistent_user_gets_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    admin = test_user(username="adm", role="admin")["user"]
    response = client.get(
        f"/api/admin/users/{uuid.uuid4()}", headers=auth_headers(admin)
    )
    assert response.status_code == 404
