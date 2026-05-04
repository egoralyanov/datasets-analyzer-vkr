"""
Интеграционные тесты API анализа: запуск, polling, получение результата,
безопасность user_id, обработка ошибок.

Особенность TestClient: BackgroundTasks выполняются СИНХРОННО (это
поведение httpx-транспорта starlette). Поэтому к моменту получения ответа
202 фоновая задача уже отработала, и в БД статус = done. В реальности
через uvicorn/браузер polling работает асинхронно.
"""
from __future__ import annotations

import io
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import settings
from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult


def _clean_csv_bytes(n: int = 60) -> bytes:
    """Маленький чистый датасет с явным числовым target."""
    rows = ["a,b,target"]
    for i in range(n):
        rows.append(f"{i},{i*2},{i % 2}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _upload_csv(
    client: TestClient,
    headers: dict[str, str],
    content: bytes = None,
    filename: str = "data.csv",
) -> dict[str, Any]:
    """Хелпер: загружает CSV и возвращает тело ответа (с id, storage_path и т.д.)."""
    response = client.post(
        "/api/datasets/upload",
        headers=headers,
        files={"file": (filename, content or _clean_csv_bytes(), "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Запуск анализа
# ──────────────────────────────────────────────────────────────────────────────


def test_start_analysis_returns_202_with_correct_metadata(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)

    response = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={"target_column": "target"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["dataset_id"] == dataset["id"]
    assert body["target_column"] == "target"
    # В TestClient BG выполняется синхронно — к моменту ответа уже done.
    # Допускаем оба варианта (pending/done) на случай будущих изменений.
    assert body["status"] in {"pending", "done"}


def test_start_analysis_without_target_succeeds(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Анализ без target_column — для задач кластеризации, валиден."""
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)
    response = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={},
    )
    assert response.status_code == 202
    assert response.json()["target_column"] is None


def test_start_analysis_with_invalid_target_returns_400(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)
    response = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={"target_column": "nonexistent_column"},
    )
    assert response.status_code == 400
    assert "nonexistent_column" in response.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Получение статуса и результата (owner)
# ──────────────────────────────────────────────────────────────────────────────


def test_get_analysis_for_owner_returns_done(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)
    start = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={"target_column": "target"},
    )
    analysis_id = start.json()["id"]

    response = client.get(f"/api/analyses/{analysis_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    # BG-задача уже отработала к этому моменту.
    assert body["status"] == "done"
    assert body["finished_at"] is not None
    assert body["error_message"] is None


def test_get_result_returns_meta_features_and_flags(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)
    start = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={"target_column": "target"},
    )
    analysis_id = start.json()["id"]

    response = client.get(f"/api/analyses/{analysis_id}/result", headers=headers)
    assert response.status_code == 200
    body = response.json()
    # Структура ответа.
    assert "analysis" in body and "meta_features" in body and "flags" in body
    assert body["analysis"]["status"] == "done"
    # Meta-features содержат ключевые поля.
    meta = body["meta_features"]
    assert meta["n_rows"] == 60
    assert meta["n_cols"] == 3
    assert "dtype_counts" in meta
    assert "distributions" in meta
    # Flags — список (может быть пустым на чистых данных).
    assert isinstance(body["flags"], list)
    # Если есть флаги — у каждого правильная структура.
    for flag in body["flags"]:
        assert {"rule_code", "severity", "rule_name", "message", "context"} <= flag.keys()
        assert flag["severity"] in {"info", "warning", "critical"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Безопасность user_id (критические тесты)
# ──────────────────────────────────────────────────────────────────────────────


def test_start_analysis_for_other_user_dataset_returns_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """User B не может запустить анализ на чужом датасете."""
    user_a = test_user()["user"]
    user_b = test_user()["user"]
    dataset_a = _upload_csv(client, auth_headers(user_a))
    response = client.post(
        f"/api/datasets/{dataset_a['id']}/analyze",
        headers=auth_headers(user_b),
        json={"target_column": "target"},
    )
    # 404 (не 403) — не палим факт существования чужого датасета.
    assert response.status_code == 404
    assert response.json()["detail"] == "Датасет не найден"


def test_get_analysis_for_other_user_returns_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """User B не может посмотреть статус чужого анализа."""
    user_a = test_user()["user"]
    user_b = test_user()["user"]
    headers_a = auth_headers(user_a)
    dataset_a = _upload_csv(client, headers_a)
    start = client.post(
        f"/api/datasets/{dataset_a['id']}/analyze",
        headers=headers_a,
        json={"target_column": "target"},
    )
    analysis_id = start.json()["id"]
    response = client.get(f"/api/analyses/{analysis_id}", headers=auth_headers(user_b))
    assert response.status_code == 404


def test_get_result_for_other_user_returns_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """User B не может скачать результат чужого анализа."""
    user_a = test_user()["user"]
    user_b = test_user()["user"]
    headers_a = auth_headers(user_a)
    dataset_a = _upload_csv(client, headers_a)
    start = client.post(
        f"/api/datasets/{dataset_a['id']}/analyze",
        headers=headers_a,
        json={"target_column": "target"},
    )
    analysis_id = start.json()["id"]
    response = client.get(
        f"/api/analyses/{analysis_id}/result", headers=auth_headers(user_b)
    )
    assert response.status_code == 404


def test_list_analyses_returns_only_own(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """User A видит только свои анализы, чужие не подмешиваются."""
    user_a = test_user()["user"]
    user_b = test_user()["user"]
    headers_a = auth_headers(user_a)
    headers_b = auth_headers(user_b)

    # User A создаёт два анализа.
    ds_a = _upload_csv(client, headers_a)
    for _ in range(2):
        client.post(
            f"/api/datasets/{ds_a['id']}/analyze",
            headers=headers_a,
            json={"target_column": "target"},
        )

    # User B создаёт один.
    ds_b = _upload_csv(client, headers_b)
    client.post(
        f"/api/datasets/{ds_b['id']}/analyze",
        headers=headers_b,
        json={"target_column": "target"},
    )

    list_a = client.get("/api/analyses", headers=headers_a).json()
    list_b = client.get("/api/analyses", headers=headers_b).json()
    assert list_a["total"] == 2
    assert list_b["total"] == 1
    # Никаких пересечений ID.
    ids_a = {a["id"] for a in list_a["items"]}
    ids_b = {a["id"] for a in list_b["items"]}
    assert ids_a.isdisjoint(ids_b)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Обработка ошибок
# ──────────────────────────────────────────────────────────────────────────────


def test_analysis_with_corrupt_dataset_status_failed(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
    db_session: Session,
) -> None:
    """
    Если файл датасета удалён с диска перед запуском, run_analysis
    должен поймать ошибку и пометить анализ как failed.
    """
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)

    # Симулируем повреждение файла — удаляем с диска до анализа.
    Path(settings.DATASETS_DIR).joinpath(str(user.id)).joinpath(
        Path(dataset["original_filename"]).stem + ".csv"
    )  # имя на диске UUID — простоe удаление через shutil rmtree директории пользователя
    user_dir = Path(settings.DATASETS_DIR) / str(user.id)
    if user_dir.exists():
        shutil.rmtree(user_dir, ignore_errors=True)

    start = client.post(
        f"/api/datasets/{dataset['id']}/analyze",
        headers=headers,
        json={"target_column": "target"},
    )
    # Лёгкая валидация target_column тоже читает с диска и упадёт раньше — 400.
    # Поэтому отправляем без target_column, чтобы валидация не сработала и
    # анализ дошёл до BG-задачи.
    if start.status_code == 400:
        start = client.post(
            f"/api/datasets/{dataset['id']}/analyze",
            headers=headers,
            json={},
        )
    assert start.status_code == 202
    analysis_id = start.json()["id"]

    response = client.get(f"/api/analyses/{analysis_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"]


def test_get_result_for_running_analysis_returns_409(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
    db_session: Session,
) -> None:
    """
    GET /result для незавершённого анализа → 409.

    BG-задача в TestClient синхронна, поэтому нормально получить running
    через API нельзя — создаём запись напрямую в БД со статусом running.
    """
    user = test_user()["user"]
    headers = auth_headers(user)
    dataset = _upload_csv(client, headers)

    # Создаём анализ напрямую в БД со статусом running, минуя API.
    analysis = Analysis(
        dataset_id=uuid.UUID(dataset["id"]),
        user_id=user.id,
        target_column="target",
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(analysis)
    db_session.commit()

    response = client.get(
        f"/api/analyses/{analysis.id}/result", headers=headers
    )
    assert response.status_code == 409
    assert "running" in response.json()["detail"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Автовозврат подвисших running при старте приложения
# ──────────────────────────────────────────────────────────────────────────────


def test_running_analyses_recovered_to_failed_on_startup(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
    db_session: Session,
) -> None:
    """
    Прямой тест функции reset_running_to_failed: «зависший» running
    переводится в failed с понятным error_message.
    Имитируем подвисший анализ + дёргаем функцию вручную.
    """
    from app.repositories.analysis_repo import reset_running_to_failed

    user = test_user()["user"]
    dataset = _upload_csv(client, auth_headers(user))

    stuck = Analysis(
        dataset_id=uuid.UUID(dataset["id"]),
        user_id=user.id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(stuck)
    db_session.commit()
    db_session.refresh(stuck)

    recovered = reset_running_to_failed(db_session)
    assert recovered >= 1

    db_session.refresh(stuck)
    assert stuck.status == "failed"
    assert stuck.error_message == "Прервано перезапуском сервера"
    assert stuck.finished_at is not None
