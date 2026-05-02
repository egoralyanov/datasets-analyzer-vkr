"""Интеграционные тесты загрузки и управления датасетами."""
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings


def _csv_bytes() -> bytes:
    return b"a,b,c\n1,2,foo\n4,5,bar\n7,8,baz\n"


def _xlsx_bytes() -> bytes:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def test_upload_csv_success(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    response = client.post(
        "/api/datasets/upload",
        headers=auth_headers(user),
        files={"file": ("data.csv", _csv_bytes(), "text/csv")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "data.csv"
    assert body["format"] == "csv"
    assert body["n_rows"] == 3
    assert body["n_cols"] == 3
    assert body["preview"]["columns"] == ["a", "b", "c"]
    assert len(body["preview"]["rows"]) == 3
    assert body["preview"]["rows"][0] == [1, 2, "foo"]
    # dtypes — строки, не numpy-объекты
    assert all(isinstance(v, str) for v in body["preview"]["dtypes"].values())


def test_upload_xlsx_success(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    response = client.post(
        "/api/datasets/upload",
        headers=auth_headers(user),
        files={
            "file": (
                "data.xlsx",
                _xlsx_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["format"] == "xlsx"
    assert body["n_rows"] == 3
    assert body["preview"]["columns"] == ["x", "y"]


def test_upload_too_large(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Лимит размера: понижаем MAX_FILE_SIZE_MB до 1 КБ и шлём 2 КБ."""
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0.001)  # ~1 КБ
    user = test_user()["user"]
    big = b"a,b\n" + b"1,2\n" * 1000  # ~6 КБ
    response = client.post(
        "/api/datasets/upload",
        headers=auth_headers(user),
        files={"file": ("big.csv", big, "text/csv")},
    )
    assert response.status_code == 413


def test_upload_wrong_format(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    response = client.post(
        "/api/datasets/upload",
        headers=auth_headers(user),
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_without_auth(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("data.csv", _csv_bytes(), "text/csv")},
    )
    assert response.status_code == 401


def test_get_someone_elses_dataset(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Чужой датасет должен возвращать 404 (не 403)."""
    alice = test_user(username="alice")["user"]
    bob = test_user(username="bob")["user"]

    upload = client.post(
        "/api/datasets/upload",
        headers=auth_headers(alice),
        files={"file": ("alice.csv", _csv_bytes(), "text/csv")},
    )
    assert upload.status_code == 201
    alice_dataset_id = upload.json()["id"]

    response = client.get(
        f"/api/datasets/{alice_dataset_id}",
        headers=auth_headers(bob),
    )
    assert response.status_code == 404


def test_delete_own_dataset(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    upload = client.post(
        "/api/datasets/upload",
        headers=auth_headers(user),
        files={"file": ("data.csv", _csv_bytes(), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["id"]

    # Берём storage_path напрямую из БД, чтобы потом проверить отсутствие файла.
    from app.core.db import SessionLocal
    from app.models.dataset import Dataset

    with SessionLocal() as db:
        from sqlalchemy import select
        ds = db.scalar(select(Dataset).where(Dataset.id == dataset_id))
        assert ds is not None
        storage_path = ds.storage_path

    assert Path(storage_path).exists()

    response = client.delete(
        f"/api/datasets/{dataset_id}",
        headers=auth_headers(user),
    )
    assert response.status_code == 204
    assert not Path(storage_path).exists()
