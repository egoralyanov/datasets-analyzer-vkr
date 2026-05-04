"""
Unit-тесты ORM-модели Report.

Покрывают:
- server_default='pending' и NULL-able поля при создании пустого Report.
- Каскадное удаление при `DELETE analysis` (ondelete=CASCADE на analysis_id).
- Каскадное удаление при `DELETE user` (ondelete=CASCADE на user_id).
- Наличие composite-индекса (user_id, status) на уровне БД через `inspect`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from app.core.db import engine
from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User


def _make_dataset(db: Session, user: User) -> Dataset:
    """Создаёт Dataset с минимально валидным набором обязательных полей."""
    dataset = Dataset(
        user_id=user.id,
        original_filename="titanic.csv",
        storage_path=f"/data/datasets/{user.id}/test.csv",
        file_size_bytes=1024,
        format="csv",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def _make_analysis(db: Session, user: User, dataset: Dataset) -> Analysis:
    """Создаёт Analysis в pending для указанного user/dataset."""
    analysis = Analysis(
        dataset_id=dataset.id,
        user_id=user.id,
        target_column=None,
        status="pending",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def test_report_defaults(
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    """Создание Report без явных полей: status=pending, NULL-able поля = None."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset)

    report = Report(analysis_id=analysis.id, user_id=user.id)
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)

    assert report.id is not None
    assert report.status == "pending"
    assert report.file_path is None
    assert report.file_size_bytes is None
    assert report.error is None
    assert report.created_at is not None
    assert report.updated_at is not None


def test_report_cascade_on_analysis_delete(
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    """DELETE analysis → связанный report исчезает (FK ondelete=CASCADE)."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset)

    report = Report(analysis_id=analysis.id, user_id=user.id, status="success")
    db_session.add(report)
    db_session.commit()
    report_id = report.id

    db_session.delete(analysis)
    db_session.commit()

    assert db_session.get(Report, report_id) is None


def test_report_cascade_on_user_delete(
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    """DELETE user → каскад уносит analyses → reports."""
    user = test_user()["user"]
    dataset = _make_dataset(db_session, user)
    analysis = _make_analysis(db_session, user, dataset)

    report = Report(analysis_id=analysis.id, user_id=user.id, status="running")
    db_session.add(report)
    db_session.commit()
    report_id = report.id

    db_session.delete(user)
    db_session.commit()

    assert db_session.get(Report, report_id) is None


def test_report_user_status_composite_index_present() -> None:
    """Composite-индекс ix_reports_user_status существует в БД на полях (user_id, status)."""
    inspector = inspect(engine)
    indexes = inspector.get_indexes("reports")
    index_by_name = {idx["name"]: idx for idx in indexes}
    assert "ix_reports_user_status" in index_by_name
    assert index_by_name["ix_reports_user_status"]["column_names"] == ["user_id", "status"]
