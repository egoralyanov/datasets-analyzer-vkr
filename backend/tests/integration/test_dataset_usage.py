"""
Интеграционные тесты GET /api/datasets/{id}/usage (Спринт 6, Phase 4.2).

Контракт:
- 200 + {analyses_count, reports_count} — владелец или admin.
- 401 — без авторизации.
- 404 — чужой/несуществующий dataset для не-админа; несуществующий — для admin.

`reports_count` считает только PDF-отчёты со status='success' — failed/pending
не включены.
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


def test_usage_owner_sees_counts(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Владелец видит счётчики: 2 анализа, 1 success-отчёт (failed не считается)."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    a1 = _make_analysis(db_session, user, dataset, status="done")
    a2 = _make_analysis(db_session, user, dataset, status="done")
    _make_report(db_session, user, a1, status="success")
    _make_report(db_session, user, a2, status="failed")

    response = client.get(
        f"/api/datasets/{dataset.id}/usage",
        headers=auth_headers(user),
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"analyses_count": 2, "reports_count": 1}


def test_usage_zero_counts(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Свежий датасет — нули."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)

    response = client.get(
        f"/api/datasets/{dataset.id}/usage",
        headers=auth_headers(user),
    )
    assert response.status_code == 200
    assert response.json() == {"analyses_count": 0, "reports_count": 0}


def test_usage_admin_sees_counts_for_other_user(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Admin видит usage чужого датасета без 404."""
    owner = test_user(username="owner")["user"]
    admin = test_user(username="adminuser", role="admin")["user"]

    dataset = _make_dataset(db_session, owner)
    a1 = _make_analysis(db_session, owner, dataset)
    _make_report(db_session, owner, a1, status="success")

    response = client.get(
        f"/api/datasets/{dataset.id}/usage",
        headers=auth_headers(admin),
    )
    assert response.status_code == 200
    assert response.json() == {"analyses_count": 1, "reports_count": 1}


def test_usage_other_user_gets_404(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Чужой датасет для не-админа → 404 (не палим существование)."""
    alice = test_user(username="alice")["user"]
    bob = test_user(username="bob")["user"]

    dataset = _make_dataset(db_session, alice)

    response = client.get(
        f"/api/datasets/{dataset.id}/usage",
        headers=auth_headers(bob),
    )
    assert response.status_code == 404


def test_usage_unauthenticated_gets_401(client: TestClient) -> None:
    """Без Authorization-заголовка → 401."""
    response = client.get(f"/api/datasets/{uuid.uuid4()}/usage")
    assert response.status_code == 401


def test_usage_nonexistent_gets_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Несуществующий dataset_id → 404 как для пользователя, так и для admin."""
    user = test_user()["user"]
    admin = test_user(username="adm2", role="admin")["user"]
    fake_id = uuid.uuid4()

    user_resp = client.get(
        f"/api/datasets/{fake_id}/usage", headers=auth_headers(user)
    )
    assert user_resp.status_code == 404

    admin_resp = client.get(
        f"/api/datasets/{fake_id}/usage", headers=auth_headers(admin)
    )
    assert admin_resp.status_code == 404
