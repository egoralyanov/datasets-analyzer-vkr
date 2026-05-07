"""
Integration-тесты GET /api/me/stats (Sprint 6, Phase 7).

Покрытие:
- 401 без auth.
- Свежий пользователь без артефактов → нули по всем четырём счётчикам.
- Пользователь с датасетами/анализами/отчётами → корректные счётчики;
  failed/pending анализы и failed-отчёты не попадают в успешные.
- Scoping: артефакты другого пользователя не учитываются.
"""
from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report


def _add_dataset(db_session: Session, user_id, idx: int) -> Dataset:
    dataset = Dataset(
        user_id=user_id,
        original_filename=f"d{idx}.csv",
        storage_path=f"/data/datasets/{user_id}/d{idx}.csv",
        file_size_bytes=1024,
        file_hash=secrets.token_hex(32),
        format="csv",
        n_rows=10,
        n_cols=3,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def _add_analysis(
    db_session: Session, dataset: Dataset, status: str
) -> Analysis:
    analysis = Analysis(
        dataset_id=dataset.id,
        user_id=dataset.user_id,
        target_column=None,
        status=status,
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


def _add_report(db_session: Session, analysis: Analysis, status: str) -> Report:
    report = Report(
        analysis_id=analysis.id,
        user_id=analysis.user_id,
        status=status,
        file_path=(
            f"/data/reports/{analysis.user_id}/r-{secrets.token_hex(4)}.pdf"
            if status == "success"
            else None
        ),
        file_size_bytes=2048 if status == "success" else None,
    )
    db_session.add(report)
    db_session.commit()
    db_session.refresh(report)
    return report


def test_me_stats_unauthenticated_returns_401(client: TestClient) -> None:
    """Без Authorization-заголовка → 401."""
    response = client.get("/api/me/stats")
    assert response.status_code == 401


def test_me_stats_zero_counts_for_new_user(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """Свежезарегистрированный юзер без артефактов → все четыре счётчика 0."""
    bundle = test_user()
    headers = auth_headers(bundle["user"])

    response = client.get("/api/me/stats", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "datasets_count": 0,
        "analyses_count": 0,
        "successful_analyses_count": 0,
        "reports_count": 0,
    }


def test_me_stats_authenticated_returns_counts(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """
    3 датасета, 4 анализа (2 done, 1 failed, 1 running), 3 отчёта (2 success,
    1 failed) → datasets=3, analyses=4, successful=2, reports=2.
    """
    bundle = test_user()
    user = bundle["user"]
    headers = auth_headers(user)

    ds_a = _add_dataset(db_session, user.id, 0)
    ds_b = _add_dataset(db_session, user.id, 1)
    ds_c = _add_dataset(db_session, user.id, 2)

    an_done_1 = _add_analysis(db_session, ds_a, "done")
    an_done_2 = _add_analysis(db_session, ds_b, "done")
    _add_analysis(db_session, ds_c, "failed")
    _add_analysis(db_session, ds_a, "running")

    _add_report(db_session, an_done_1, "success")
    _add_report(db_session, an_done_2, "success")
    _add_report(db_session, an_done_1, "failed")

    response = client.get("/api/me/stats", headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "datasets_count": 3,
        "analyses_count": 4,
        "successful_analyses_count": 2,
        "reports_count": 2,
    }


def test_me_stats_does_not_count_other_users(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """
    Артефакты другого пользователя не должны попадать в /me/stats —
    запросы scoped по current_user.id.
    """
    alice = test_user()
    bob = test_user()

    # Алиса: 1 датасет + 1 done-анализ + 1 success-отчёт.
    alice_ds = _add_dataset(db_session, alice["user"].id, 0)
    alice_an = _add_analysis(db_session, alice_ds, "done")
    _add_report(db_session, alice_an, "success")

    # Боб: 2 датасета + 2 анализа (один done, один failed) + 1 success-отчёт.
    bob_ds_1 = _add_dataset(db_session, bob["user"].id, 1)
    bob_ds_2 = _add_dataset(db_session, bob["user"].id, 2)
    bob_an_done = _add_analysis(db_session, bob_ds_1, "done")
    _add_analysis(db_session, bob_ds_2, "failed")
    _add_report(db_session, bob_an_done, "success")

    alice_resp = client.get(
        "/api/me/stats", headers=auth_headers(alice["user"])
    ).json()
    assert alice_resp == {
        "datasets_count": 1,
        "analyses_count": 1,
        "successful_analyses_count": 1,
        "reports_count": 1,
    }

    bob_resp = client.get(
        "/api/me/stats", headers=auth_headers(bob["user"])
    ).json()
    assert bob_resp == {
        "datasets_count": 2,
        "analyses_count": 2,
        "successful_analyses_count": 1,
        "reports_count": 1,
    }
