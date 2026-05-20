"""
Интеграционные тесты для DELETE /api/datasets/{id}.

После правок 2026-05-20 удаление датасета НЕ уносит связанные анализы и
PDF-отчёты. FK `analyses.dataset_id` теперь ON DELETE SET NULL; имя файла
и размеры остаются в денормализованном снапшоте `analyses.dataset_filename`
и т.п. Старое cascade-поведение отключено по запросу комиссии: пользователь
хочет освободить место/убрать датасет из списка, но история анализов и
сгенерированных отчётов должна сохраниться.

Контракт DELETE по-прежнему 204 и скоуп по владельцу. Меняется только
семантика побочных эффектов.
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
        # Снапшот пишется обычно при создании анализа через API;
        # здесь дублируем для реалистичности.
        dataset_filename=dataset.original_filename,
        dataset_format=dataset.format,
        dataset_n_rows=dataset.n_rows,
        dataset_n_cols=dataset.n_cols,
        dataset_file_size_bytes=dataset.file_size_bytes,
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


def test_delete_dataset_keeps_associated_analyses(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Анализы остаются в БД после удаления датасета; dataset_id обнуляется,
    снапшот dataset_filename продолжает быть осмысленным."""
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
    for aid in (a1_id, a2_id):
        a = db_session.scalar(select(Analysis).where(Analysis.id == aid))
        assert a is not None
        assert a.dataset_id is None
        assert a.dataset_filename == "iris.csv"


def test_delete_dataset_keeps_associated_pdfs_on_disk(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """
    Главный гарант нового поведения: PDF success-отчётов и их записи в БД
    остаются доступными после удаления датасета. Только сам файл датасета
    с диска удаляется.
    """
    user = test_user()["user"]
    dataset, dataset_path = _make_dataset_with_file(db_session, user)
    a1 = _make_analysis(db_session, user, dataset)
    a2 = _make_analysis(db_session, user, dataset)
    _, pdf1_path = _make_report_with_file(db_session, user, a1)
    _, pdf2_path = _make_report_with_file(db_session, user, a2)

    dataset_id = dataset.id
    a1_id, a2_id = a1.id, a2.id

    assert dataset_path.exists()
    assert pdf1_path.exists()
    assert pdf2_path.exists()

    response = client.delete(
        f"/api/datasets/{dataset_id}", headers=auth_headers(user)
    )
    assert response.status_code == 204

    db_session.expire_all()
    # Файл датасета снят с диска.
    assert not dataset_path.exists()
    # PDF success-отчётов на диске сохранились.
    assert pdf1_path.exists()
    assert pdf2_path.exists()
    # Записи Report тоже живы — на них теперь висят «осиротевшие» анализы.
    reports = list(
        db_session.scalars(
            select(Report).where(Report.analysis_id.in_((a1_id, a2_id)))
        )
    )
    assert len(reports) == 2


def test_delete_dataset_with_no_analyses_succeeds(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Smoke: датасет без анализов удаляется штатно."""
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
