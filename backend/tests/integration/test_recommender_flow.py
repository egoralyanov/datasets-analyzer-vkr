"""
Integration-тесты потока рекомендации типа задачи и подбора похожих.

Покрывают:
1. Iris через API → MULTICLASS_CLASSIFICATION с confidence ≥ 0.7.
2. Анализ без target → CLUSTERING.
3. Реалистичный «Titanic-like» CSV с константной колонкой (триггер NaN
   в correlation_matrix) — проверка что _jsonb_safe не валит запись и
   embedding длиной 128 сохраняется.
4. /similar?top_k=5 — 5 элементов, у каждого title/source/distance.

См. .knowledge/methods/recommender-rules.md, recommender-ml.md, dataset-matching.md.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrame → CSV bytes для multipart-upload."""
    return df.to_csv(index=False).encode("utf-8")


def _upload_and_analyze(
    client: TestClient,
    headers: dict[str, str],
    df: pd.DataFrame,
    *,
    target_column: str | None,
    filename: str = "data.csv",
) -> dict[str, Any]:
    """Загружает CSV, запускает анализ, возвращает analysis-объект (status уже done
    в TestClient — BG-задача синхронна)."""
    upload = client.post(
        "/api/datasets/upload",
        headers=headers,
        files={"file": (filename, _df_to_csv_bytes(df), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    payload: dict[str, Any] = {}
    if target_column is not None:
        payload["target_column"] = target_column

    start = client.post(
        f"/api/datasets/{dataset_id}/analyze",
        headers=headers,
        json=payload,
    )
    assert start.status_code == 202, start.text
    return start.json()


def _titanic_like_df(n: int = 60) -> pd.DataFrame:
    """
    Синтетика, имитирующая Titanic: бинарный target Survived + смешанные
    числовые/категориальные признаки + одна КОНСТАНТНАЯ колонка.

    Константная колонка — главное: pandas.corr() для константы возвращает NaN,
    что попадает в meta_features.correlation_matrix и без _jsonb_safe валит
    INSERT в JSONB.
    """
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        {
            "Pclass": rng.integers(1, 4, size=n),
            "Sex": rng.choice(["male", "female"], size=n),
            "Age": rng.uniform(0.5, 80.0, size=n).round(1),
            "Fare": rng.exponential(scale=20.0, size=n).round(2),
            # Константная колонка: provoque NaN в Pearson r → проверка _jsonb_safe.
            "Embarked": ["S"] * n,
            "Survived": rng.integers(0, 2, size=n),
        }
    )
    return df


def test_iris_analysis_recommends_multiclass(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Iris через API → MULTICLASS_CLASSIFICATION с confidence ≥ 0.7."""
    user = test_user()["user"]
    headers = auth_headers(user)
    iris_df = load_iris(as_frame=True).frame  # 4 числовых + 'target' (int 0/1/2)

    analysis = _upload_and_analyze(
        client, headers, iris_df, target_column="target", filename="iris.csv"
    )
    response = client.get(
        f"/api/analyses/{analysis['id']}/result", headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    rec = body["task_recommendation"]
    assert rec is not None, "task_recommendation should not be None on Iris"
    assert rec["task_type_code"] == "MULTICLASS_CLASSIFICATION"
    assert rec["confidence"] >= 0.7
    # Слой 1 справится сам — через рулы, без делегирования в ML.
    assert rec["source"] in {"rules", "hybrid"}


def test_no_target_recommends_clustering(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """Анализ без target_column → ветка 1 правил → CLUSTERING."""
    user = test_user()["user"]
    headers = auth_headers(user)
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {f"f{i}": rng.normal(size=80) for i in range(5)}
    )

    analysis = _upload_and_analyze(
        client, headers, df, target_column=None, filename="no_target.csv"
    )
    response = client.get(
        f"/api/analyses/{analysis['id']}/result", headers=headers
    )
    assert response.status_code == 200
    rec = response.json()["task_recommendation"]
    assert rec is not None
    assert rec["task_type_code"] == "CLUSTERING"


def test_titanic_like_embedding_not_null_and_size_128(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """
    «Titanic-like» CSV с константной колонкой Embarked='S' проверяет,
    что _jsonb_safe корректно санитизирует NaN в correlation_matrix
    (pandas.corr на константе → NaN), и embedding пишется как vector(128).
    """
    user = test_user()["user"]
    headers = auth_headers(user)

    analysis = _upload_and_analyze(
        client,
        headers,
        _titanic_like_df(n=60),
        target_column="Survived",
        filename="titanic_like.csv",
    )
    response = client.get(
        f"/api/analyses/{analysis['id']}/result", headers=headers
    )
    assert response.status_code == 200, response.text
    body = response.json()
    embedding = body["embedding"]
    assert embedding is not None, "embedding should be set when scaler available"
    assert len(embedding) == 128
    assert all(isinstance(v, (int, float)) for v in embedding)
    # Sanity: sklearn ставит таргет как BINARY на 0/1
    assert body["task_recommendation"]["task_type_code"] == "BINARY_CLASSIFICATION"


def test_get_similar_returns_5_with_distance(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    """GET /similar?top_k=5 → ровно 5 записей с distance (>=0) и базовыми полями."""
    user = test_user()["user"]
    headers = auth_headers(user)
    iris_df = load_iris(as_frame=True).frame

    analysis = _upload_and_analyze(
        client, headers, iris_df, target_column="target", filename="iris.csv"
    )
    response = client.get(
        f"/api/analyses/{analysis['id']}/similar?top_k=5", headers=headers
    )
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) == 5
    expected_keys = {
        "id",
        "title",
        "description",
        "source",
        "source_url",
        "task_type_code",
        "distance",
    }
    for item in items:
        assert expected_keys <= set(item.keys())
        assert isinstance(item["distance"], (int, float))
        assert item["distance"] >= 0.0
        assert item["title"]
        assert item["task_type_code"]
