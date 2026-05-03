"""
Unit-тесты сервиса baseline_trainer.

Тесты бьют только train_baseline_from_df (без БД) — это изолирует логику
обучения от плавающих интеграционных грабель, не требует фикстуры
PostgreSQL и проходит за единицы секунд. БД-обёртка train_baseline
покрывается интеграционными тестами в Phase 8.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris

from app.services.baseline_trainer import _preprocess, train_baseline_from_df


def _build_meta(
    cardinality_by_column: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Минимальный meta-словарь, достаточный для baseline-препроцессинга."""
    return {"cardinality_by_column": cardinality_by_column or {}}


def test_iris_multiclass_accuracy_above_0_9() -> None:
    """RandomForest на Iris (4 числовых признака, 3 класса) должен давать
    accuracy_mean >= 0.9 — sanity-проверка корректности связки препроцессинга,
    StratifiedKFold и scoring='accuracy'."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # колонки: 4 признака + 'target' (int 0/1/2)

    result = train_baseline_from_df(
        df=df,
        meta=_build_meta(),
        leakage_cols=[],
        target_col="target",
        task_type="MULTICLASS_CLASSIFICATION",
    )

    rf_acc_mean = result["metrics"]["random_forest"]["accuracy"]["mean"]
    assert rf_acc_mean >= 0.9, f"RF accuracy на Iris упала до {rf_acc_mean:.3f}"
    assert result["models"] == ["logistic_regression", "random_forest"]
    assert result["n_rows_used"] == 150
    assert result["n_features_used"] == 4


def test_diabetes_regression_r2_above_0_3() -> None:
    """RandomForestRegressor с зафиксированными лимитами на Diabetes даёт
    r2_mean ≈ 0.4-0.45; нижняя граница 0.3 — запас на флуктуации фолдов."""
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame  # 10 числовых признаков + 'target' (continuous)

    result = train_baseline_from_df(
        df=df,
        meta=_build_meta(),
        leakage_cols=[],
        target_col="target",
        task_type="REGRESSION",
    )

    rf_r2_mean = result["metrics"]["random_forest"]["r2"]["mean"]
    assert rf_r2_mean >= 0.3, f"RF r2 на Diabetes упал до {rf_r2_mean:.3f}"
    assert result["models"] == ["ridge", "random_forest"]
    # Регрессионные метрики содержат mae и rmse в человеческом виде
    # (преобразование из neg_mean_*), а не sklearn-овские neg_*.
    assert "mae" in result["metrics"]["random_forest"]
    assert "rmse" in result["metrics"]["random_forest"]


def test_leakage_columns_excluded() -> None:
    """Колонки из leakage_cols не попадают в feature_importance и присутствуют
    в excluded_columns_due_to_leakage. Сценарий: 'leak' буквально равна target —
    без исключения RF выучил бы её и feature_importance показала бы 100% на ней."""
    rng = np.random.default_rng(seed=42)
    n = 300
    target = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            # Точная копия target → если бы 'leak' попала в X, RF дал бы ей 100%.
            "leak": target.astype(int),
            "target": target,
        }
    )

    result = train_baseline_from_df(
        df=df,
        meta=_build_meta(),
        leakage_cols=["leak"],
        target_col="target",
        task_type="BINARY_CLASSIFICATION",
    )

    assert "leak" not in result["feature_importance"], (
        "Колонка с подозрением на утечку просочилась в feature_importance"
    )
    assert "leak" in result["excluded_columns_due_to_leakage"]
    # x1/x2 — чистый шум, обе должны быть в importance после исключения leak.
    assert set(result["feature_importance"].keys()) <= {"x1", "x2"}


def test_preprocess_handles_nan() -> None:
    """_preprocess не падает на NaN в числовых и категориальных колонках,
    итоговый X не содержит пропусков (числовые → медиана, категориальные → мода)."""
    rng = np.random.default_rng(seed=42)
    n = 200

    num_with_nan = rng.normal(size=n)
    num_with_nan[::10] = np.nan  # ~10% NaN

    cat_values = np.array(["a", "b", "c"])[rng.integers(0, 3, size=n)]
    cat_with_nan: np.ndarray = np.array(cat_values, dtype=object)
    cat_with_nan[::15] = None  # ~7% None

    df = pd.DataFrame(
        {
            "num": num_with_nan,
            "cat": cat_with_nan,
            "target": rng.integers(0, 2, size=n),
        }
    )
    # ratio = 3/200 = 0.015 < 0.1 → 'cat' попадает в one-hot.
    meta = _build_meta(cardinality_by_column={"cat": 3 / n})

    X, y = _preprocess(
        df,
        target_col="target",
        leakage_cols=[],
        meta=meta,
        task_type="BINARY_CLASSIFICATION",
    )

    assert X.isna().sum().sum() == 0, "X содержит NaN после препроцессинга"
    assert len(y) == len(X), "Длины X и y разошлись"
    # 'cat' должна быть закодирована в 3 dummy-колонки.
    assert any(col.startswith("cat_") for col in X.columns), (
        "One-hot encoding не применился к низкокардинальной 'cat'"
    )


def test_clustering_returns_stub_without_crash() -> None:
    """Для CLUSTERING возвращается стуб с пустыми моделями и непустым note —
    обучение не запускается, сторонние эффекты не вызываются."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    result = train_baseline_from_df(
        df=df,
        meta=_build_meta(),
        leakage_cols=[],
        target_col="a",
        task_type="CLUSTERING",
    )

    assert result["models"] == []
    assert result["metrics"] == {}
    assert result["feature_importance"] == {}
    assert "note" in result and result["note"]
    assert "trained_at" in result
