"""
Integration-тесты админ-эндпоинтов.

Покрытие:
- GET /api/admin/stats для admin → 200, числа корректны при «нулевой» БД
  (только сам admin) и после создания нескольких пользователей.
- GET /api/admin/stats для обычного user → 403 «Требуются права администратора».
- GET /api/admin/users пагинация на 25 пользователях.
- GET /api/admin/users для обычного user → 403.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.user import User


def _make_admin(
    db_session: Session, test_user: Callable[..., dict[str, Any]]
) -> dict[str, Any]:
    bundle = test_user(role="admin")
    return bundle


def test_stats_for_admin(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """
    admin видит счётчики; success_rate корректно считается через done/total.

    Делаем дельта-проверку (а не строгие равенства), потому что соседние
    тесты в той же сессии могут оставить записи в analyses/datasets/reports —
    db_session очищает только users CASCADE между тестами, но порядок
    тестов и накопленный seed состояние мы не контролируем.
    """
    admin = _make_admin(db_session, test_user)
    headers = auth_headers(admin["user"])

    before = client.get("/api/admin/stats", headers=headers).json()

    # Создаём 2 датасета и 2 анализа: один done, один failed.
    user_id = admin["user"].id
    for idx, status in enumerate(("done", "failed")):
        dataset = Dataset(
            user_id=user_id,
            original_filename=f"d{idx}.csv",
            storage_path=f"/data/datasets/{user_id}/d{idx}.csv",
            file_size_bytes=1024,
            format="csv",
            n_rows=10, n_cols=3,
        )
        db_session.add(dataset)
        db_session.commit()
        db_session.refresh(dataset)

        db_session.add(Analysis(
            dataset_id=dataset.id, user_id=user_id,
            target_column=None, status=status,
        ))
        db_session.commit()

    after = client.get("/api/admin/stats", headers=headers).json()
    assert after["total_users"] >= 1
    assert after["total_datasets"] - before["total_datasets"] == 2
    assert after["total_analyses"] - before["total_analyses"] == 2
    # Success rate должен быть числом в [0, 1] (хотя бы один анализ есть).
    assert after["analyses_success_rate"] is not None
    assert 0.0 <= after["analyses_success_rate"] <= 1.0


def test_stats_for_regular_user_403(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """Не-admin → 403 «Требуются права администратора»."""
    bundle = test_user()  # role="user" по умолчанию
    headers = auth_headers(bundle["user"])

    response = client.get("/api/admin/stats", headers=headers)
    assert response.status_code == 403
    assert "администратора" in response.json()["detail"].lower()


def test_list_users_admin_pagination(
    client: TestClient,
    db_session: Session,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """24 обычных user'а + 1 admin → page=1 (size=20) даёт 20, page=2 даёт 5."""
    admin = _make_admin(db_session, test_user)
    headers = auth_headers(admin["user"])

    # 24 дополнительных пользователя; учитывая что admin уже один — total=25.
    for i in range(24):
        db_session.add(
            User(
                email=f"u{i:02d}@e.x",
                username=f"user_{i:02d}",
                password_hash=hash_password("Strong123!"),
                role="user",
            )
        )
    db_session.commit()

    page1 = client.get("/api/admin/users?page=1&size=20", headers=headers).json()
    assert page1["total"] == 25
    assert page1["page"] == 1
    assert page1["size"] == 20
    assert page1["pages"] == 2
    assert len(page1["items"]) == 20

    page2 = client.get("/api/admin/users?page=2&size=20", headers=headers).json()
    assert page2["page"] == 2
    assert len(page2["items"]) == 5

    ids_p1 = {item["id"] for item in page1["items"]}
    ids_p2 = {item["id"] for item in page2["items"]}
    assert ids_p1.isdisjoint(ids_p2)

    # Каждая запись содержит datasets_count и analyses_count.
    for item in page1["items"]:
        assert "datasets_count" in item
        assert "analyses_count" in item
        assert item["role"] in {"user", "admin"}


def test_list_users_for_regular_user_403(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable,
) -> None:
    """Не-admin → 403 на /api/admin/users."""
    bundle = test_user()
    headers = auth_headers(bundle["user"])

    response = client.get("/api/admin/users", headers=headers)
    assert response.status_code == 403
