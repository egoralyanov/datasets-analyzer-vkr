"""
Интеграционные тесты для расширенного DELETE /api/datasets/{id}
(Спринт 6, Phase 4.6).

Закрывают known limitation #5 из Sprint 4: при удалении датасета записи
analyses / reports уходят через FK cascade, но PDF-файлы оставались
осиротевшими на диске. Теперь они тоже удаляются.

Контракт DELETE /api/datasets/{id} не менялся (тот же 204, тот же скоуп
по владельцу). Только внутренняя логика расширена: snapshot путей до
delete + unlink после commit, OSError логируется WARNING без отката БД.

Базовые сценарии delete (404, 401, успех с одним файлом датасета) уже
покрыты в test_datasets_upload.py — здесь только новые ветки cascade.
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


def _make_report_with_file(
    db: Session, user: User, analysis: Analysis
) -> tuple[Report, Path]:
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


def test_delete_dataset_removes_associated_analyses(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """После DELETE дочерние анализы ушли (FK cascade). Тест фиксирует поведение."""
    user = test_user()["user"]
    dataset, _ = _make_dataset_with_file(db_session, user)
    a1 = _make_analysis(db_session, user, dataset)
    a2 = _make_analysis(db_session, user, dataset)
    a1_id, a2_id, dataset_id = a1.id, a2.id, dataset.id

    response = client.delete(
        f"/api/datasets/{dataset_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204

    db_session.expire_all()
    assert db_session.scalar(
        select(Dataset).where(Dataset.id == dataset_id)
    ) is None
    assert db_session.scalar(select(Analysis).where(Analysis.id == a1_id)) is None
    assert db_session.scalar(select(Analysis).where(Analysis.id == a2_id)) is None


def test_delete_dataset_removes_associated_pdfs_from_disk(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """
    Главный тест ради которого затевалось 4.6: PDF success-отчёта всех
    анализов датасета исчезает с диска после удаления датасета.
    Также проверяет, что failed-отчёт без файла на диске не ломает поток.
    """
    user = test_user()["user"]
    dataset, dataset_path = _make_dataset_with_file(db_session, user)
    a1 = _make_analysis(db_session, user, dataset)
    a2 = _make_analysis(db_session, user, dataset)
    _, pdf1_path = _make_report_with_file(db_session, user, a1)
    _, pdf2_path = _make_report_with_file(db_session, user, a2)

    # failed-report — без физического файла, проверяем что он не ломает unlink-цикл.
    failed_report = Report(
        analysis_id=a2.id,
        user_id=user.id,
        status="failed",
        file_path=None,
        error="boom",
    )
    db_session.add(failed_report)
    db_session.commit()

    # Сохраняем id-шники до expire_all, иначе ORM при последующем доступе
    # к атрибутам уже удалённых instances кидает ObjectDeletedError.
    dataset_id = dataset.id
    a1_id, a2_id = a1.id, a2.id

    assert dataset_path.exists()
    assert pdf1_path.exists()
    assert pdf2_path.exists()

    response = client.delete(
        f"/api/datasets/{dataset_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204

    # Cascade и unlink сработали по обоим артефактам.
    db_session.expire_all()
    assert db_session.scalar(
        select(Report).where(Report.analysis_id.in_((a1_id, a2_id)))
    ) is None
    assert not dataset_path.exists()
    assert not pdf1_path.exists()
    assert not pdf2_path.exists()


def test_delete_dataset_with_no_analyses_succeeds(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Smoke: датасет без анализов удаляется как раньше — пустой report_paths не ломает поток."""
    user = test_user()["user"]
    dataset, dataset_path = _make_dataset_with_file(db_session, user)
    dataset_id = dataset.id

    assert dataset_path.exists()

    response = client.delete(
        f"/api/datasets/{dataset_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204

    db_session.expire_all()
    assert db_session.scalar(
        select(Dataset).where(Dataset.id == dataset_id)
    ) is None
    assert not dataset_path.exists()
