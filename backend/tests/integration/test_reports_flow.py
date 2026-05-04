"""
Integration-тесты потока генерации PDF-отчёта.

Покрытие:
- POST /api/analyses/{id}/report — happy path, owner-скоуп, конфликты 409.
- GET /api/reports/{id} — текущий статус.
- GET /api/reports/{id}/download — PDF-выдача, оба класса 404 (running и
  missing-on-disk), запрет на чужой отчёт.

Особенность TestClient: BackgroundTasks выполняются СИНХРОННО после
возврата response. По умолчанию заменяем `generate_report` на no-op
через monkeypatch — каждый тест иначе пытался бы поднять matplotlib
+ WeasyPrint на пустой БД-цепочке. E2E через настоящий рендер уже
покрывается test_report_service.test_generate_report_happy_path.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api import reports as reports_api
from app.config import settings
from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.dataset import Dataset
from app.models.report import Report


@pytest.fixture(autouse=True)
def _stub_generate_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет generate_report no-op'ом во всём этом тестовом модуле."""
    monkeypatch.setattr(reports_api, "generate_report", lambda *a, **kw: None)


@pytest.fixture
def completed_analysis(
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """
    Фабрика «всё готово к отчёту»: пользователь, токен, datasets+analyses
    в done, минимальный AnalysisResult с meta_features.

    Возвращает dict {user, password, token, headers, analysis_id, dataset_id}.
    """
    from app.core.security import create_access_token

    def _make() -> dict[str, Any]:
        bundle = test_user()
        user = bundle["user"]
        token = create_access_token(user.id, user.role)
        headers = {"Authorization": f"Bearer {token}"}

        dataset = Dataset(
            user_id=user.id,
            original_filename="iris.csv",
            storage_path=f"/data/datasets/{user.id}/iris.csv",
            file_size_bytes=2048,
            format="csv",
            n_rows=150,
            n_cols=5,
        )
        db_session.add(dataset)
        db_session.commit()
        db_session.refresh(dataset)

        analysis = Analysis(
            dataset_id=dataset.id,
            user_id=user.id,
            target_column="species",
            status="done",
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)

        result = AnalysisResult(
            analysis_id=analysis.id,
            meta_features={"n_rows": 150, "n_cols": 5},
        )
        db_session.add(result)
        db_session.commit()

        return {
            "user": user,
            "password": bundle["password"],
            "token": token,
            "headers": headers,
            "analysis_id": str(analysis.id),
            "dataset_id": str(dataset.id),
        }

    return _make


# ──────────────────────────────────────────────────────────────────────────
# POST /api/analyses/{id}/report
# ──────────────────────────────────────────────────────────────────────────


def test_create_report_success(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
) -> None:
    ctx = completed_analysis()
    response = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    report_id = uuid.UUID(body["id"])

    db_session.expire_all()
    saved = db_session.get(Report, report_id)
    assert saved is not None
    assert saved.user_id == ctx["user"].id
    assert str(saved.analysis_id) == ctx["analysis_id"]
    assert saved.status == "pending"


def test_create_report_when_running_returns_409(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
) -> None:
    ctx = completed_analysis()
    first = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    assert first.status_code == 202
    first_id = first.json()["id"]

    second = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    assert second.status_code == 409
    body = second.json()
    assert body["reason"] == "report_in_progress"
    assert body["report_id"] == first_id
    assert body["status"] == "pending"


def test_create_report_for_others_analysis_returns_404(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    ctx = completed_analysis()
    intruder = test_user()["user"]
    intruder_headers = auth_headers(intruder)
    response = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=intruder_headers
    )
    assert response.status_code == 404


def test_create_report_when_analysis_not_done_returns_409(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
) -> None:
    ctx = completed_analysis()
    # Возвращаем analysis в pending — отчёт по нему пока запрашивать нельзя.
    analysis = db_session.get(Analysis, uuid.UUID(ctx["analysis_id"]))
    assert analysis is not None
    analysis.status = "pending"
    db_session.commit()

    response = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "analysis_not_done"
    assert body.get("report_id") is None


# ──────────────────────────────────────────────────────────────────────────
# GET /api/reports/{id}
# ──────────────────────────────────────────────────────────────────────────


def test_get_report_status_returns_current_state(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
) -> None:
    ctx = completed_analysis()
    create = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    report_id = create.json()["id"]

    pending_resp = client.get(f"/api/reports/{report_id}", headers=ctx["headers"])
    assert pending_resp.status_code == 200
    assert pending_resp.json()["status"] == "pending"

    # Эмулируем завершение работы фон-задачи: руками меняем статус в БД.
    report = db_session.get(Report, uuid.UUID(report_id))
    assert report is not None
    report.status = "success"
    report.file_size_bytes = 2048
    db_session.commit()

    success_resp = client.get(f"/api/reports/{report_id}", headers=ctx["headers"])
    assert success_resp.status_code == 200
    payload = success_resp.json()
    assert payload["status"] == "success"
    assert payload["file_size_bytes"] == 2048
    assert payload["error"] is None


# ──────────────────────────────────────────────────────────────────────────
# GET /api/reports/{id}/download
# ──────────────────────────────────────────────────────────────────────────


def _mark_success_with_file(
    db_session: Session, report_id: str, user_id: uuid.UUID
) -> Path:
    """Помечает отчёт как success и кладёт минимальный валидный PDF на диск."""
    report = db_session.get(Report, uuid.UUID(report_id))
    assert report is not None
    relative_path = f"{user_id}/{report.id}.pdf"
    pdf_path = Path(settings.REPORTS_DIR) / relative_path
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Минимально-валидный PDF: сигнатура + EOF. Полный 14-байтный PDF
    # достаточен для проверки Content-Type и сигнатуры в downloader'е.
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    report.status = "success"
    report.file_path = relative_path
    report.file_size_bytes = pdf_path.stat().st_size
    db_session.commit()
    return pdf_path


def test_download_success_returns_pdf(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
) -> None:
    ctx = completed_analysis()
    create = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    report_id = create.json()["id"]
    pdf_path = _mark_success_with_file(db_session, report_id, ctx["user"].id)
    try:
        response = client.get(
            f"/api/reports/{report_id}/download", headers=ctx["headers"]
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF-")
        # RFC 5987: оба варианта filename должны присутствовать.
        disposition = response.headers["content-disposition"]
        assert 'filename="report.pdf"' in disposition
        assert "filename*=UTF-8''" in disposition
        assert "report_iris_" in disposition  # стем + дата
    finally:
        pdf_path.unlink(missing_ok=True)
        pdf_path.parent.rmdir()


def test_download_when_running_returns_404(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
) -> None:
    ctx = completed_analysis()
    create = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    report_id = create.json()["id"]
    response = client.get(
        f"/api/reports/{report_id}/download", headers=ctx["headers"]
    )
    assert response.status_code == 404


def test_download_when_file_missing_on_disk_returns_404(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
) -> None:
    """status=success, но файл удалили вручную → 404, не 500."""
    ctx = completed_analysis()
    create = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    report_id = create.json()["id"]
    pdf_path = _mark_success_with_file(db_session, report_id, ctx["user"].id)
    pdf_path.unlink()  # симулируем повреждённый volume

    response = client.get(
        f"/api/reports/{report_id}/download", headers=ctx["headers"]
    )
    assert response.status_code == 404
    pdf_path.parent.rmdir()


def test_download_for_others_report_returns_404(
    client: TestClient,
    completed_analysis: Callable[..., dict[str, Any]],
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    ctx = completed_analysis()
    create = client.post(
        f"/api/analyses/{ctx['analysis_id']}/report", headers=ctx["headers"]
    )
    report_id = create.json()["id"]
    pdf_path = _mark_success_with_file(db_session, report_id, ctx["user"].id)
    try:
        intruder = test_user()["user"]
        intruder_headers = auth_headers(intruder)
        response = client.get(
            f"/api/reports/{report_id}/download", headers=intruder_headers
        )
        # 404, не 403 — паттерн «не палим существование».
        assert response.status_code == 404
    finally:
        pdf_path.unlink(missing_ok=True)
        pdf_path.parent.rmdir()
