"""
Unit-тесты для профайлера (backend/app/services/profiler.py).

Каждый тест работает на синтетических данных (numpy + pandas), не зависит
от файловой системы и БД. Цель — зафиксировать корректность математических
утверждений из .knowledge/methods/profiling.md, чтобы их можно было защитить
на ГЭК ссылкой на конкретный зелёный тест.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.profiler import (
    SAMPLE_SIZE,
    SAMPLING_THRESHOLD,
    compute_entropy,
    compute_kurtosis,
    compute_meta_features,
    compute_mi_with_target,
    compute_skewness,
    detect_outliers_iqr,
    detect_outliers_zscore,
    is_normal_shapiro,
    maybe_sample,
    normalized_entropy,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Outliers
# ──────────────────────────────────────────────────────────────────────────────


def test_iqr_detects_obvious_outliers() -> None:
    """IQR-метод Тьюки находит очевидный выброс в массиве [1..5, 100]."""
    values = np.array([1, 2, 3, 4, 5, 100], dtype=float)
    mask = detect_outliers_iqr(values)
    assert mask[-1] is np.True_ or mask[-1]  # последнее значение — выброс
    assert mask[:-1].sum() == 0  # остальные — нет


def test_iqr_no_outliers_in_normal_data() -> None:
    """На большой нормальной выборке IQR-метод даёт ≤ 5% выбросов."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 1000)
    mask = detect_outliers_iqr(values)
    # Для нормального распределения IQR помечает ~0.7% точек как выбросы.
    assert mask.sum() / values.size < 0.05


def test_zscore_threshold_3_marks_only_extremes() -> None:
    """Z-score с порогом 3 помечает только значения с |z| > 3."""
    rng = np.random.default_rng(42)
    # 200 точек из N(0,1) + один явный выброс на 10 единиц от среднего.
    values = np.concatenate([rng.normal(0, 1, 200), np.array([10.0])])
    mask = detect_outliers_zscore(values, threshold=3.0)
    # Последний элемент (10.0) однозначно за пределами 3σ.
    assert mask[-1]
    # Среди исходных 200 нормальных — ни одного выброса не должно быть.
    assert mask[:-1].sum() == 0


def test_zscore_zero_std_returns_no_outliers() -> None:
    """На константном массиве Z-score не должен делить на ноль (граничный случай)."""
    values = np.full(10, 7.0)
    mask = detect_outliers_zscore(values)
    assert mask.sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Distribution: Shapiro-Wilk и моменты
# ──────────────────────────────────────────────────────────────────────────────


def test_shapiro_normal_passes() -> None:
    """На нормальной выборке Шапиро-Уилк не отвергает H0 (p > 0.05)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 500)
    is_normal, p = is_normal_shapiro(values)
    assert is_normal is True
    assert p is not None and p > 0.05


def test_shapiro_skewed_fails() -> None:
    """На сильно асимметричной (экспоненциальной) выборке тест отвергает H0."""
    rng = np.random.default_rng(42)
    values = rng.exponential(1.0, 500)
    is_normal, p = is_normal_shapiro(values)
    assert is_normal is False
    assert p is not None and p < 0.05


def test_shapiro_constant_returns_not_normal() -> None:
    """Граничный случай: на константе тест не должен падать с warning."""
    values = np.full(10, 5.0)
    is_normal, p = is_normal_shapiro(values)
    assert is_normal is False and p is None


def test_skewness_zero_for_normal() -> None:
    """Для нормального распределения skewness ≈ 0 (с допуском ~0.2)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 5000)
    skew = compute_skewness(values)
    assert skew is not None and abs(skew) < 0.2


def test_skewness_positive_for_exponential() -> None:
    """Для экспоненциального распределения skewness > 1 (теоретически 2)."""
    rng = np.random.default_rng(42)
    values = rng.exponential(1.0, 5000)
    skew = compute_skewness(values)
    assert skew is not None and skew > 1.0


def test_kurtosis_near_zero_for_normal() -> None:
    """Для нормального эксцесс по Фишеру ≈ 0 (теоретически точно 0)."""
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 5000)
    kurt = compute_kurtosis(values)
    assert kurt is not None and abs(kurt) < 0.3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Information theory: энтропия, MI
# ──────────────────────────────────────────────────────────────────────────────


def test_entropy_zero_for_constant() -> None:
    """H(X) = 0 для константного распределения (всё одинаково)."""
    series = pd.Series(["a"] * 100)
    assert compute_entropy(series) == 0.0


def test_entropy_max_for_uniform() -> None:
    """H(X) ≈ log₂(k) для равномерного распределения по k категориям."""
    series = pd.Series(["a", "b", "c", "d"] * 25)
    h = compute_entropy(series)
    assert h is not None and abs(h - np.log2(4)) < 1e-6


def test_normalized_entropy_one_for_uniform() -> None:
    """Нормированная энтропия равномерного распределения равна 1.0."""
    series = pd.Series(["a", "b", "c"] * 30)
    assert normalized_entropy(series) == pytest.approx(1.0, abs=1e-6)


def test_mi_target_none_returns_null() -> None:
    """Без target Mutual Information не определена — возвращаем None."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert compute_mi_with_target(df, None) is None


def test_mi_perfect_correlation_high() -> None:
    """y = f(x) → MI(x, y) высокая (близка к энтропии y)."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.integers(0, 4, n)
    df = pd.DataFrame({"x": x, "y": x})  # точная копия
    result = compute_mi_with_target(df, "y")
    assert result is not None and result["max"] > 1.0  # log₂(4) = 2 бит


def test_mi_independent_low() -> None:
    """Независимые признаки → MI ≈ 0."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 500),
            "y": rng.choice([0, 1], 500),
        }
    )
    result = compute_mi_with_target(df, "y")
    assert result is not None and result["max"] < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 4. Сэмплирование
# ──────────────────────────────────────────────────────────────────────────────


def test_sampling_not_triggered_below_threshold() -> None:
    """При len(df) ≤ SAMPLING_THRESHOLD выборка возвращается без изменений."""
    df = pd.DataFrame({"a": np.arange(100)})
    sampled, info = maybe_sample(df)
    assert info["sampled"] is False
    assert len(sampled) == 100


def test_sampling_triggered_above_50k() -> None:
    """При len(df) > SAMPLING_THRESHOLD происходит сэмплирование до SAMPLE_SIZE."""
    rng = np.random.default_rng(42)
    n = SAMPLING_THRESHOLD + 10_000
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.choice([0, 1], n)})
    sampled, info = maybe_sample(df, target_col="y")
    assert info["sampled"] is True
    assert info["original_size"] == n
    # Стратифицированный сэмпл может отличаться на ±1 строку из-за округления.
    assert abs(len(sampled) - SAMPLE_SIZE) < 100


# ──────────────────────────────────────────────────────────────────────────────
# 5. Главная функция compute_meta_features
# ──────────────────────────────────────────────────────────────────────────────


def _make_titanic_like(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Синтетический Titanic-подобный датасет для интеграционных проверок."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(30, 10, n),
            "fare": rng.exponential(20, n),
            "sex": rng.choice(["m", "f"], n),
            "pclass": rng.choice(["1", "2", "3"], n, p=[0.2, 0.3, 0.5]),
            "survived": rng.choice([0, 1], n, p=[0.6, 0.4]),
        }
    )


def test_compute_meta_features_returns_full_set() -> None:
    """На синтетическом датасете 200×5 возвращаются все ключевые поля meta-features."""
    df = _make_titanic_like()
    meta = compute_meta_features(df, target_col="survived")
    expected_keys = {
        "n_rows",
        "n_cols",
        "dtype_counts",
        "memory_mb",
        "total_missing_pct",
        "max_col_missing_pct",
        "duplicate_rows_pct",
        "mean_skewness",
        "mean_kurtosis",
        "outliers_pct",
        "normality_test_pvalue",
        "high_cardinality_cols",
        "low_variance_numeric_cols",
        "low_variance_categorical_cols",
        "target_kind",
        "target_imbalance_ratio",
        "target_class_entropy",
        "max_abs_correlation",
        "target_correlation_max",
        "target_mutual_information_max",
        "distributions",
        "sampling",
    }
    missing = expected_keys - set(meta.keys())
    assert not missing, f"missing meta keys: {missing}"
    assert meta["n_rows"] == 200
    assert meta["n_cols"] == 5


def test_compute_meta_features_no_target_target_fields_null() -> None:
    """Без target_col все target_*-поля должны быть строго None (явно null)."""
    df = _make_titanic_like()
    meta = compute_meta_features(df, target_col=None)
    assert meta["target_kind"] is None
    assert meta["target_imbalance_ratio"] is None
    assert meta["target_class_entropy"] is None
    assert meta["target_skewness"] is None
    assert meta["target_mutual_information_max"] is None
    assert meta["target_correlation_max"] is None


def test_compute_meta_features_dtype_counts_correct() -> None:
    """dtype_counts корректно группирует колонки по их pandas dtype."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.0, 2.0, 3.0],
            "c": ["x", "y", "z"],
        }
    )
    meta = compute_meta_features(df)
    assert meta["dtype_counts"]["int64"] == 1
    assert meta["dtype_counts"]["float64"] == 1
    # В pandas 3.x dtype для строк — "str" (раньше "object"); проверяем,
    # что нечисловая колонка хотя бы попала в один из категориальных типов.
    string_count = meta["dtype_counts"].get("str", 0) + meta["dtype_counts"].get(
        "object", 0
    )
    assert string_count == 1
    # Сумма всех типов должна равняться количеству колонок.
    assert sum(meta["dtype_counts"].values()) == 3


def test_compute_meta_features_duplicates_detected() -> None:
    """duplicate_rows_pct корректно отражает повторяющиеся строки."""
    base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = pd.concat([base, base, base], ignore_index=True)
    meta = compute_meta_features(df)
    # Из 9 строк уникальных 3 → 6 дубликатов = 6/9 ≈ 0.667.
    assert meta["duplicate_rows_pct"] == pytest.approx(6 / 9, rel=1e-3)
