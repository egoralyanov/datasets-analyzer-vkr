"""
Unit-тесты orchestrator'а PDF-отчёта.

Структура: один e2e через `generate_report` (включая WeasyPrint),
четыре быстрых через `_render_html` напрямую (без WeasyPrint), один на
failure-path с моком `_render_pdf_bytes`.

Реалистичные meta_features конструируются inline по схеме профайлера
(distributions.numeric / categorical / correlation_matrix /
target_value_counts / target_kind / прочее).
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.core.db import SessionLocal
from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.dataset import Dataset
from app.models.report import Report
from app.models.user import User
from app.services import report_service


# ──────────────────────────────────────────────────────────────────────────
# Фикстуры
# ──────────────────────────────────────────────────────────────────────────


def _iris_like_meta() -> dict[str, Any]:
    """Сжатый Iris-подобный meta_features для прогона рендера end-to-end."""
    return {
        "n_rows": 150,
        "n_cols": 5,
        "dtype_counts": {"float64": 4, "object": 1},
        "total_missing_pct": 0.0,
        "max_col_missing_pct": 0.0,
        "missing_by_column": {
            "sepal_length": 0.0, "sepal_width": 0.0,
            "petal_length": 0.0, "petal_width": 0.0, "species": 0.0,
        },
        "target_kind": "categorical",
        "target_n_unique": 3,
        "target_imbalance_ratio": 1.0,
        "target_class_entropy": 1.585,
        "target_skewness": None,
        "target_value_counts": {"setosa": 50, "versicolor": 50, "virginica": 50},
        "correlation_matrix": {
            "sepal_length": {"sepal_length": 1.0, "sepal_width": -0.12,
                              "petal_length": 0.87, "petal_width": 0.82},
            "sepal_width":  {"sepal_length": -0.12, "sepal_width": 1.0,
                              "petal_length": -0.43, "petal_width": -0.37},
            "petal_length": {"sepal_length": 0.87, "sepal_width": -0.43,
                              "petal_length": 1.0, "petal_width": 0.96},
            "petal_width":  {"sepal_length": 0.82, "sepal_width": -0.37,
                              "petal_length": 0.96, "petal_width": 1.0},
        },
        "distributions": {
            "numeric": {
                "sepal_length": {
                    "bin_edges": [4.3, 4.9, 5.5, 6.1, 6.7, 7.3, 7.9],
                    "counts": [9, 32, 41, 42, 19, 7],
                },
                "petal_length": {
                    "bin_edges": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    "counts": [37, 13, 11, 25, 42, 22],
                },
            },
            "categorical": {
                "species": {
                    "categories": ["setosa", "versicolor", "virginica"],
                    "counts": [50, 50, 50],
                    "other_count": 0,
                },
            },
        },
    }


def _minimal_context(**overrides: Any) -> dict[str, Any]:
    """
    Собирает context для `_render_html` без обращения к БД.

    Дефолты — пустые/нейтральные. Тесты переопределяют только нужные ключи.
    """
    base = {
        "title": "Отчёт",
        "generated_at": "2026-05-04 12:00 UTC",
        "report_id": str(uuid.uuid4()),
        "user": {"username": "tester"},
        "dataset": {"original_filename": "data.csv", "format": "csv"},
        "analysis": {
            "id": str(uuid.uuid4()),
            "target_column": "target",
            "started_at": None,
            "finished_at": None,
        },
        "summary": {
            "n_rows": 100, "n_cols": 5,
            "total_missing_pct": 0.0, "max_col_missing_pct": 0.0,
            "dtype_counts": {"float64": 5},
        },
        "target_info": {
            "kind": "categorical", "n_unique": 2,
            "imbalance_ratio": 1.0, "class_entropy": 1.0, "skewness": None,
        },
        "quality_flags": [],
        "task_recommendation": None,
        "similar_datasets": [],
        "baseline": None,
        "charts": {
            "numeric_histograms": [],
            "categorical_bars": [],
            "correlation_heatmap": None,
            "target_chart": None,
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def report_chain(
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
) -> Callable[..., Report]:
    """
    Фабрика полной цепочки User → Dataset → Analysis → AnalysisResult → Report
    в БД. По умолчанию Iris-meta, без baseline и без embedding.
    """

    def _make(
        *,
        meta: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
    ) -> Report:
        user = test_user()["user"]
        dataset = Dataset(
            user_id=user.id,
            original_filename="iris.csv",
            storage_path=f"/data/datasets/{user.id}/iris.csv",
            file_size_bytes=4_096,
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
            status="success",
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)

        result = AnalysisResult(
            analysis_id=analysis.id,
            meta_features=meta if meta is not None else _iris_like_meta(),
            embedding=None,
            task_recommendation={
                "task_type_code": "MULTICLASS_CLASSIFICATION",
                "confidence": 0.95,
                "source": "rules",
                "applied_rules": ["CATEGORICAL_TARGET_LOW_CARDINALITY"],
                "explanation": "Категориальный target с 3 уникальными классами.",
                "ml_probabilities": None,
            },
            baseline=baseline,
        )
        db_session.add(result)
        db_session.commit()

        report = Report(analysis_id=analysis.id, user_id=user.id, status="pending")
        db_session.add(report)
        db_session.commit()
        db_session.refresh(report)
        return report

    return _make


# ──────────────────────────────────────────────────────────────────────────
# Тесты
# ──────────────────────────────────────────────────────────────────────────


def test_generate_report_happy_path(
    report_chain: Callable[..., Report],
    db_session: Session,
) -> None:
    """E2E через WeasyPrint: PDF создаётся на диске, запись success, размер > 0."""
    report = report_chain()
    report_id = report.id
    user_id = report.user_id

    report_service.generate_report(report_id, SessionLocal)

    # После завершения фон-задачи перезагружаем запись в db_session.
    db_session.expire_all()
    saved = db_session.get(Report, report_id)
    assert saved is not None
    assert saved.status == "success", f"unexpected status={saved.status} error={saved.error!r}"
    assert saved.file_path == f"{user_id}/{report_id}.pdf"
    assert saved.file_size_bytes is not None and saved.file_size_bytes > 1024
    assert saved.error is None

    pdf_path = Path(settings.REPORTS_DIR) / saved.file_path
    assert pdf_path.exists()
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) == saved.file_size_bytes

    # Чистим за собой созданный файл, чтобы не накапливать в data/reports/
    # между прогонами.
    pdf_path.unlink(missing_ok=True)
    pdf_path.parent.rmdir()


def test_render_html_skips_missing_baseline() -> None:
    """Без baseline шаблон не рендерит секцию «Базовая модель»."""
    html = report_service._render_html(_minimal_context(baseline=None))
    assert "Базовая модель" not in html
    assert "feature-importance-bar" not in html


def test_render_html_skips_missing_similar() -> None:
    """Без similar_datasets секция «Похожие датасеты» не появляется."""
    html = report_service._render_html(_minimal_context(similar_datasets=[]))
    assert "Похожие датасеты" not in html


def test_render_html_handles_long_dataset_name() -> None:
    """200-символьное имя файла не ломает шаблон и попадает в HTML целиком."""
    long_name = "x" * 200
    ctx = _minimal_context(
        title=f"Отчёт по анализу датасета «{long_name}»",
        dataset={"original_filename": long_name, "format": "csv"},
    )
    html = report_service._render_html(ctx)
    assert long_name in html
    assert html.startswith("<!DOCTYPE html>")


def test_render_html_replaces_none_meta_with_dash() -> None:
    """target_column=None и пустые типы → шаблон выводит дефис, не «None»."""
    ctx = _minimal_context(
        analysis={
            "id": str(uuid.uuid4()),
            "target_column": None,
            "started_at": None,
            "finished_at": None,
        },
        target_info={
            "kind": None, "n_unique": None,
            "imbalance_ratio": None, "class_entropy": None, "skewness": None,
        },
    )
    html = report_service._render_html(ctx)
    # Поле «Целевая переменная» должно быть с тире.
    assert "<td>—</td>" in html
    # Литерал «None» в шаблоне не должен появляться.
    assert ">None<" not in html


def test_generate_report_failure_records_status_and_error(
    report_chain: Callable[..., Report],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сбой в _render_pdf_bytes → status=failed, error обрезан до 1000 символов."""
    report = report_chain()
    report_id = report.id

    long_message = "boom-" + "x" * 2000  # явно длиннее 1000

    def _raise(*_args: Any, **_kwargs: Any) -> bytes:
        raise RuntimeError(long_message)

    monkeypatch.setattr(report_service, "_render_pdf_bytes", _raise)

    report_service.generate_report(report_id, SessionLocal)

    db_session.expire_all()
    saved = db_session.get(Report, report_id)
    assert saved is not None
    assert saved.status == "failed"
    assert saved.error is not None
    assert len(saved.error) == report_service.ERROR_MAX_LEN == 1000
    assert saved.error.startswith("boom-")
    assert saved.file_path is None
    assert saved.file_size_bytes is None
