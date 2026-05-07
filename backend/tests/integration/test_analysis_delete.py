"""
Интеграционные тесты DELETE /api/analyses/{id} (Спринт 6, Phase 4.3).

Контракт:
- 204 — успешное удаление (владелец или admin).
- 401 — без auth.
- 404 — чужой/несуществующий analysis для не-админа; несуществующий — для admin.
- 409 — running profiling / running baseline (плоское тело {"detail": "..."}).

Каскад: при удалении analysis из БД уходят analysis_results, quality_flags
и reports (FK ondelete=CASCADE). PDF-файлы success-отчётов удаляются с
диска отдельно.
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
from app.models.analysis_result import AnalysisResult
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


def _make_result(
    db: Session,
    analysis: Analysis,
    *,
    baseline_status: str = "not_started",
) -> AnalysisResult:
    result = AnalysisResult(
        analysis_id=analysis.id,
        meta_features={"n_rows": 150, "n_cols": 5},
        baseline_status=baseline_status,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def _make_report_with_file(
    db: Session, user: User, analysis: Analysis
) -> tuple[Report, Path]:
    """Создаёт success-Report + физический файл на диске; возвращает (report, abs_path)."""
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


def test_delete_owner_succeeds_204(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset, status="done")

    analysis_id = analysis.id
    response = client.delete(
        f"/api/analyses/{analysis_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204
    # SELECT мимо identity map (которая держит уже удалённый instance из
    # эндпоинт-сессии): запись действительно ушла из БД.
    db_session.expire_all()
    assert db_session.scalar(
        select(Analysis).where(Analysis.id == analysis_id)
    ) is None


def test_delete_admin_succeeds_for_other_user(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    owner = test_user(username="owner")["user"]
    admin = test_user(username="adm", role="admin")["user"]
    dataset = _make_dataset(db_session, owner)
    analysis = _make_analysis(db_session, owner, dataset, status="done")

    analysis_id = analysis.id
    response = client.delete(
        f"/api/analyses/{analysis_id}", headers=auth_headers(admin)
    )
    assert response.status_code == 204
    db_session.expire_all()
    assert db_session.scalar(
        select(Analysis).where(Analysis.id == analysis_id)
    ) is None


def test_delete_other_user_gets_404(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Чужой анализ для не-админа → 404 (не палим существование)."""
    alice = test_user(username="alice")["user"]
    bob = test_user(username="bob")["user"]
    dataset = _make_dataset(db_session, alice)
    analysis = _make_analysis(db_session, alice, dataset, status="done")

    response = client.delete(
        f"/api/analyses/{analysis.id}", headers=auth_headers(bob)
    )
    assert response.status_code == 404
    # Запись не тронута.
    assert db_session.get(Analysis, analysis.id) is not None


def test_delete_unauthenticated_gets_401(client: TestClient) -> None:
    response = client.delete(f"/api/analyses/{uuid.uuid4()}")
    assert response.status_code == 401


def test_delete_nonexistent_gets_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    response = client.delete(
        f"/api/analyses/{uuid.uuid4()}", headers=auth_headers(user)
    )
    assert response.status_code == 404


def test_delete_removes_pdf_file_from_disk(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """После 204 PDF success-отчёта на диске больше нет."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset, status="done")
    _, pdf_path = _make_report_with_file(db_session, user, analysis)
    assert pdf_path.exists()
    analysis_id = analysis.id

    response = client.delete(
        f"/api/analyses/{analysis_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204

    # Cascade унёс запись Report; файл удалён эндпоинтом.
    db_session.expire_all()
    assert db_session.scalar(
        select(Report).where(Report.analysis_id == analysis_id)
    ) is None
    assert not pdf_path.exists()

    # Cleanup parent dir for hygiene.
    try:
        pdf_path.parent.rmdir()
    except OSError:
        pass


def test_delete_with_running_profiling_returns_409(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """analysis.status='running' → 409 + плоский detail про profiling."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset, status="running")

    response = client.delete(
        f"/api/analyses/{analysis.id}", headers=auth_headers(user)
    )
    assert response.status_code == 409
    body = response.json()
    assert isinstance(body["detail"], str)
    assert "profiling" in body["detail"].lower()
    # Запись НЕ удалена.
    assert db_session.get(Analysis, analysis.id) is not None


def test_delete_with_running_baseline_returns_409(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """analysis.status='done' + result.baseline_status='running' → 409 про baseline."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset, status="done")
    _make_result(db_session, analysis, baseline_status="running")

    response = client.delete(
        f"/api/analyses/{analysis.id}", headers=auth_headers(user)
    )
    assert response.status_code == 409
    body = response.json()
    assert isinstance(body["detail"], str)
    assert "baseline" in body["detail"].lower()
    assert db_session.get(Analysis, analysis.id) is not None
