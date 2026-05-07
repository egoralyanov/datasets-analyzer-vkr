"""
Интеграционные тесты DELETE /api/admin/users/{id} (Спринт 6, Phase 4.4).

Контракт:
- 204 — успех (cascade убирает датасеты, анализы, отчёты + файлы с диска).
- 401 — без auth.
- 403 — auth есть, но не admin.
- 404 — несуществующий user_id.
- 409 — самоудаление, последний admin.

Бизнес-проверки в порядке: 404 → self → last-admin → cascade.

Формат 409 — плоский `{"detail": "..."}`.
"""
from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User


def _make_dataset_with_file(
    db: Session, user: User
) -> tuple[Dataset, Path]:
    """Dataset + физический файл на диске, чтобы проверить cleanup."""
    storage_dir = Path(settings.DATASETS_DIR) / str(user.id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{uuid.uuid4()}.csv"
    storage_path.write_bytes(b"a,b\n1,2\n")
    dataset = Dataset(
        user_id=user.id,
        original_filename="iris.csv",
        storage_path=str(storage_path),
        file_size_bytes=storage_path.stat().st_size,
        file_hash=secrets.token_hex(32),
        format="csv",
        n_rows=1,
        n_cols=2,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset, storage_path


def _make_analysis(
    db: Session, user: User, dataset: Dataset
) -> Analysis:
    analysis = Analysis(
        dataset_id=dataset.id,
        user_id=user.id,
        target_column=None,
        status="done",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _make_report_with_file(
    db: Session, user: User, analysis: Analysis
) -> tuple[Report, Path]:
    """success-Report + физический PDF; возвращает (report, abs_path)."""
    relative = f"{user.id}/{uuid.uuid4()}.pdf"
    abs_path = Path(settings.REPORTS_DIR) / relative
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"%PDF-1.4 fake")
    report = Report(
        analysis_id=analysis.id,
        user_id=user.id,
        status="success",
        file_path=relative,
        file_size_bytes=abs_path.stat().st_size,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report, abs_path


def test_admin_deletes_user_succeeds_204(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Базовый happy path: admin удаляет обычного пользователя без контента."""
    admin = test_user(username="adm", role="admin")["user"]
    target = test_user(username="victim")["user"]
    target_id = target.id

    response = client.delete(
        f"/api/admin/users/{target_id}", headers=auth_headers(admin)
    )
    assert response.status_code == 204

    db_session.expire_all()
    assert db_session.scalar(select(User).where(User.id == target_id)) is None


def test_admin_deletes_self_returns_409(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Самоудаление — 409, даже когда в системе ещё один admin (срабатывает self, не last)."""
    admin1 = test_user(username="admin1", role="admin")["user"]
    test_user(username="admin2", role="admin")

    response = client.delete(
        f"/api/admin/users/{admin1.id}", headers=auth_headers(admin1)
    )
    assert response.status_code == 409
    body = response.json()
    assert isinstance(body["detail"], str)
    assert "your own admin account" in body["detail"].lower()


def test_admin_deletes_last_admin_via_self_returns_409(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """
    Один admin в системе, удаляет себя → срабатывает self-check (он первым в
    порядке), detail — про own account, а не про last admin. Это ожидаемое
    поведение: self-check точнее описывает реальную ситуацию.
    """
    admin = test_user(username="onlyadmin", role="admin")["user"]

    response = client.delete(
        f"/api/admin/users/{admin.id}", headers=auth_headers(admin)
    )
    assert response.status_code == 409
    body = response.json()
    assert "your own admin account" in body["detail"].lower()


def test_user_deletes_other_user_returns_403(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Не-admin → 403 (это /api/admin/*, существование сущности не секрет)."""
    plain_user = test_user(username="plain")["user"]
    target = test_user(username="other")["user"]

    response = client.delete(
        f"/api/admin/users/{target.id}", headers=auth_headers(plain_user)
    )
    assert response.status_code == 403


def test_unauthenticated_returns_401(client: TestClient) -> None:
    response = client.delete(f"/api/admin/users/{uuid.uuid4()}")
    assert response.status_code == 401


def test_delete_nonexistent_returns_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    admin = test_user(username="adm", role="admin")["user"]
    response = client.delete(
        f"/api/admin/users/{uuid.uuid4()}", headers=auth_headers(admin)
    )
    assert response.status_code == 404


def test_cascade_removes_datasets_and_analyses_and_files(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Полный happy-path: удаление юзера убирает датасет/анализ/отчёт + файлы с диска."""
    admin = test_user(username="adm", role="admin")["user"]
    victim = test_user(username="vict")["user"]
    victim_id = victim.id

    dataset, dataset_path = _make_dataset_with_file(db_session, victim)
    analysis = _make_analysis(db_session, victim, dataset)
    report, pdf_path = _make_report_with_file(db_session, victim, analysis)
    dataset_id, analysis_id, report_id = dataset.id, analysis.id, report.id

    assert dataset_path.exists()
    assert pdf_path.exists()

    response = client.delete(
        f"/api/admin/users/{victim_id}", headers=auth_headers(admin)
    )
    assert response.status_code == 204

    db_session.expire_all()
    # БД-уровень: всё ушло через FK cascade.
    assert db_session.scalar(select(User).where(User.id == victim_id)) is None
    assert db_session.scalar(
        select(Dataset).where(Dataset.id == dataset_id)
    ) is None
    assert db_session.scalar(
        select(Analysis).where(Analysis.id == analysis_id)
    ) is None
    assert db_session.scalar(
        select(Report).where(Report.id == report_id)
    ) is None

    # Диск: оба файла удалены эндпоинтом после коммита.
    assert not dataset_path.exists()
    assert not pdf_path.exists()
