"""
Integration-тесты страницы «История анализов» (пагинированный
GET /api/analyses).

Покрывают:
- Постраничную выборку с непересечением ID между страницами.
- Фильтр по статусу (?status=done).
- Скоуп по пользователю (другой юзер видит только свои).
- Дефолтные параметры (без query → page=1, size=20) и подтягивание
  dataset_name через joinedload.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.dataset import Dataset


def _seed_analyses(
    db: Session,
    user_id: uuid.UUID,
    *,
    n_done: int = 0,
    n_failed: int = 0,
    dataset_name: str = "iris.csv",
    with_recommendation: bool = False,
) -> list[uuid.UUID]:
    """
    Создаёт N анализов с заданным статусом, возвращая их ID в порядке
    `started_at DESC` (соответствует ожидаемой выдаче API).
    """
    dataset = Dataset(
        user_id=user_id,
        original_filename=dataset_name,
        storage_path=f"/data/datasets/{user_id}/{dataset_name}",
        file_size_bytes=2048,
        format="csv",
        n_rows=150,
        n_cols=5,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    created: list[Analysis] = []
    for status, count in (("done", n_done), ("failed", n_failed)):
        for _ in range(count):
            analysis = Analysis(
                dataset_id=dataset.id,
                user_id=user_id,
                target_column="species",
                status=status,
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            if with_recommendation and status == "done":
                result = AnalysisResult(
                    analysis_id=analysis.id,
                    meta_features={"n_rows": 150, "n_cols": 5},
                    task_recommendation={
                        "task_type_code": "MULTICLASS_CLASSIFICATION",
                        "confidence": 0.95,
                        "source": "rules",
                        "applied_rules": [],
                        "explanation": "",
                    },
                )
                db.add(result)
                db.commit()
            created.append(analysis)

    # Возвращаем в порядке started_at DESC — самые свежие первыми.
    return [a.id for a in sorted(created, key=lambda a: a.started_at, reverse=True)]


@pytest.fixture
def authed_user(
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> Callable[..., dict[str, Any]]:
    """Фабрика «пользователь + готовые headers»."""

    def _make() -> dict[str, Any]:
        bundle = test_user()
        user = bundle["user"]
        return {"user": user, "headers": auth_headers(user)}

    return _make


def test_list_analyses_pagination(
    client: TestClient,
    db_session: Session,
    authed_user: Callable[..., dict[str, Any]],
) -> None:
    """25 анализов → page=1 (size=20) даёт 20 элементов, page=2 — оставшиеся 5; ID не пересекаются."""
    ctx = authed_user()
    ids_desc = _seed_analyses(db_session, ctx["user"].id, n_done=25)

    page1 = client.get("/api/analyses?page=1&size=20", headers=ctx["headers"]).json()
    page2 = client.get("/api/analyses?page=2&size=20", headers=ctx["headers"]).json()

    assert page1["total"] == 25
    assert page1["page"] == 1
    assert page1["size"] == 20
    assert page1["pages"] == 2
    assert len(page1["items"]) == 20
    # Самые свежие на странице 1, в правильном порядке started_at DESC.
    assert [item["id"] for item in page1["items"]] == [str(i) for i in ids_desc[:20]]

    assert page2["total"] == 25
    assert page2["page"] == 2
    assert len(page2["items"]) == 5
    assert [item["id"] for item in page2["items"]] == [str(i) for i in ids_desc[20:]]

    # Ключевая защита: ID на page=1 и page=2 не пересекаются.
    ids_p1 = {item["id"] for item in page1["items"]}
    ids_p2 = {item["id"] for item in page2["items"]}
    assert ids_p1.isdisjoint(ids_p2)


def test_list_analyses_filter_by_status(
    client: TestClient,
    db_session: Session,
    authed_user: Callable[..., dict[str, Any]],
) -> None:
    """5 done + 5 failed; ?status=done возвращает только успешные."""
    ctx = authed_user()
    _seed_analyses(db_session, ctx["user"].id, n_done=5, n_failed=5)

    only_done = client.get(
        "/api/analyses?status=done", headers=ctx["headers"]
    ).json()
    assert only_done["total"] == 5
    assert only_done["pages"] == 1
    assert all(item["status"] == "done" for item in only_done["items"])


def test_list_analyses_only_own(
    client: TestClient,
    db_session: Session,
    authed_user: Callable[..., dict[str, Any]],
) -> None:
    """Анализы другого пользователя не должны протекать в выдачу."""
    user_a = authed_user()
    user_b = authed_user()
    _seed_analyses(db_session, user_a["user"].id, n_done=3)
    _seed_analyses(db_session, user_b["user"].id, n_done=2)

    list_a = client.get("/api/analyses", headers=user_a["headers"]).json()
    list_b = client.get("/api/analyses", headers=user_b["headers"]).json()

    assert list_a["total"] == 3
    assert list_b["total"] == 2

    ids_a = {item["id"] for item in list_a["items"]}
    ids_b = {item["id"] for item in list_b["items"]}
    assert ids_a.isdisjoint(ids_b)


def test_list_analyses_default_pagination_no_params(
    client: TestClient,
    db_session: Session,
    authed_user: Callable[..., dict[str, Any]],
) -> None:
    """
    Без query-параметров — page=1, size=20 по умолчанию. dataset_name
    подтягивается через joinedload(Analysis.dataset), recommended_task_type
    — через joinedload(Analysis.result).task_recommendation.
    """
    ctx = authed_user()
    _seed_analyses(
        db_session, ctx["user"].id,
        n_done=2,
        dataset_name="titanic.csv",
        with_recommendation=True,
    )

    response = client.get("/api/analyses", headers=ctx["headers"]).json()
    assert response["page"] == 1
    assert response["size"] == 20
    assert response["total"] == 2
    assert response["pages"] == 1
    assert len(response["items"]) == 2

    item = response["items"][0]
    assert item["dataset_name"] == "titanic.csv"
    assert item["recommended_task_type"] == "MULTICLASS_CLASSIFICATION"
    assert item["status"] == "done"
    assert item["target_column"] == "species"
    # Временные метки сериализуются ISO-строкой.
    assert "started_at" in item
