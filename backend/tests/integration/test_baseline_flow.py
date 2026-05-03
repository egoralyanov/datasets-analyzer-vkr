"""
Integration-тесты потока baseline-обучения.

Покрывают:
1. POST /baseline на done-анализе → 202 + status running/done.
2. POST + polling до done → metrics и feature_importance непустые.
3. GET /baseline после done → корректная структура.
4. Безопасность: чужой анализ → 404 (паттерн Спринта 1).

Особенность TestClient: BackgroundTasks выполняются СИНХРОННО после
возврата response. Это значит к моменту получения 202 baseline уже
обучен или завершился ошибкой. Polling-логика всё равно работает:
один запрос — и сразу видим done.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _setup_done_analysis(
    client: TestClient,
    headers: dict[str, str],
) -> str:
    """Загружает Iris и запускает анализ; возвращает analysis_id (status уже done)."""
    df = load_iris(as_frame=True).frame
    upload = client.post(
        "/api/datasets/upload",
        headers=headers,
        files={"file": ("iris.csv", _df_to_csv_bytes(df), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]
    start = client.post(
        f"/api/datasets/{dataset_id}/analyze",
        headers=headers,
        json={"target_column": "target"},
    )
    assert start.status_code == 202, start.text
    return start.json()["id"]


def _wait_for_baseline_done(
    client: TestClient,
    headers: dict[str, str],
    analysis_id: str,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.2,
) -> dict[str, Any]:
    """Polling GET /baseline пока baseline_status='done' либо timeout."""
    deadline = time.monotonic() + timeout_s
    last_body: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/analyses/{analysis_id}/baseline", headers=headers)
        # До POST /baseline GET даст 404; после — 200.
        if resp.status_code == 404:
            time.sleep(poll_interval_s)
            continue
        assert resp.status_code == 200, resp.text
        last_body = resp.json()
        if last_body["baseline_status"] in {"done", "failed"}:
            return last_body
        time.sleep(poll_interval_s)
    raise AssertionError(
        f"Baseline did not reach done within {timeout_s}s; last={last_body}"
    )


def test_post_baseline_on_done_analysis_returns_202(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """POST /baseline на done-анализе → 202 + baseline_status='running'."""
    user = test_user()["user"]
    headers = auth_headers(user)
    analysis_id = _setup_done_analysis(client, headers)

    resp = client.post(
        f"/api/analyses/{analysis_id}/baseline", headers=headers
    )
    # TestClient выполняет BG синхронно, но эндпоинт всё равно возвращает 202
    # с baseline_status='running' — статус 'done' пишется только через GET после
    # завершения BG-задачи.
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["analysis_id"] == analysis_id
    assert body["baseline_status"] == "running"


def test_baseline_polling_completes(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """POST + polling → baseline_status='done', metrics и feature_importance непусты."""
    user = test_user()["user"]
    headers = auth_headers(user)
    analysis_id = _setup_done_analysis(client, headers)

    post = client.post(
        f"/api/analyses/{analysis_id}/baseline", headers=headers
    )
    assert post.status_code == 202

    body = _wait_for_baseline_done(client, headers, analysis_id)
    assert body["baseline_status"] == "done", body
    baseline = body["baseline"]
    assert baseline is not None
    assert baseline["metrics"], "metrics should not be empty"
    assert baseline["feature_importance"], "feature_importance should not be empty"


def test_get_baseline_after_done_shows_metrics(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """GET /baseline после done содержит models / metrics / feature_importance / n_rows_used."""
    user = test_user()["user"]
    headers = auth_headers(user)
    analysis_id = _setup_done_analysis(client, headers)
    client.post(f"/api/analyses/{analysis_id}/baseline", headers=headers)
    _wait_for_baseline_done(client, headers, analysis_id)

    resp = client.get(f"/api/analyses/{analysis_id}/baseline", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline_status"] == "done"
    assert body["baseline_error"] is None
    baseline = body["baseline"]
    expected_keys = {
        "models",
        "metrics",
        "feature_importance",
        "n_rows_used",
        "n_features_used",
        "trained_at",
        "excluded_columns_due_to_leakage",
    }
    assert expected_keys <= set(baseline.keys())
    # На Iris (multiclass) обучается LogReg + RF.
    assert "logistic_regression" in baseline["models"]
    assert "random_forest" in baseline["models"]
    # Hood-проверка метрик: для multiclass должна быть accuracy mean.
    rf_metrics = baseline["metrics"]["random_forest"]
    assert "accuracy" in rf_metrics
    assert "mean" in rf_metrics["accuracy"]
    assert isinstance(rf_metrics["accuracy"]["mean"], (int, float))


def test_post_baseline_on_someone_elses_analysis_returns_404(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """User B не может запустить baseline на чужом анализе. Возвращаем 404 (не 403),
    паттерн Спринта 1: не палим факт существования чужого ресурса."""
    user_a = test_user()["user"]
    user_b = test_user()["user"]
    headers_a = auth_headers(user_a)
    headers_b = auth_headers(user_b)

    analysis_id = _setup_done_analysis(client, headers_a)

    resp = client.post(
        f"/api/analyses/{analysis_id}/baseline", headers=headers_b
    )
    assert resp.status_code == 404
    # GET тоже 404 для чужого.
    resp_get = client.get(
        f"/api/analyses/{analysis_id}/baseline", headers=headers_b
    )
    assert resp_get.status_code == 404
