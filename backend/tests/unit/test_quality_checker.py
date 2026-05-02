"""
Unit-тесты для quality_checker (backend/app/services/quality_checker.py).

Каждое из 12 правил покрыто минимум двумя тестами: позитивный (срабатывает
ровно как должно) и негативный (не срабатывает на чистых данных). Тесты
работают с FlagDraft напрямую, поэтому БД-сессия не требуется — только
для интеграционного теста run_quality_checks (там используется фикстура
`db_session` из conftest).
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from app.services.profiler import compute_meta_features
from app.services.quality_checker import (
    check_date_not_parsed,
    check_duplicates,
    check_high_cardinality,
    check_high_missing,
    check_imbalance_binary,
    check_imbalance_multiclass,
    check_leakage_suspicion,
    check_low_variance,
    check_outliers,
    check_small_dataset,
    check_target_missing,
    check_too_few_features,
    run_quality_checks,
)

DEFAULT_THRESHOLDS: dict[str, dict] = {
    "TARGET_MISSING": {"max_target_missing_pct": 0.05},
    "LEAKAGE": {
        "max_correlation_with_target": 0.95,
        "max_mutual_info_with_target": 0.9,
    },
    "HIGH_MISSING": {"max_col_missing_pct": 0.3},
    "DUPLICATES": {"max_duplicates_pct": 0.05},
    "IMBALANCE_BINARY": {"max_imbalance_ratio": 10.0},
    "IMBALANCE_MULTICLASS": {"min_class_size": 50},
    "SMALL_DATASET": {"min_rows": 100},
    "TOO_FEW_FEATURES": {"min_cols": 3},
    "LOW_VARIANCE": {"min_cv": 0.01, "min_normalized_entropy": 0.1},
    "HIGH_CARDINALITY": {"max_cardinality_ratio": 0.5},
    "OUTLIERS": {"max_outliers_pct": 0.05},
    "DATE_NOT_PARSED": {"min_date_parse_rate": 0.9},
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. TARGET_MISSING (critical)
# ──────────────────────────────────────────────────────────────────────────────


def test_target_missing_above_threshold_flags() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [1, 0, None, None]})
    flags = check_target_missing(df, "target", {}, DEFAULT_THRESHOLDS["TARGET_MISSING"])
    assert len(flags) == 1 and flags[0].rule_code == "TARGET_MISSING"
    assert flags[0].context["missing_pct"] == pytest.approx(0.5)


def test_target_missing_below_threshold_no_flag() -> None:
    df = pd.DataFrame({"x": list(range(100)), "target": [1] * 99 + [None]})
    flags = check_target_missing(df, "target", {}, DEFAULT_THRESHOLDS["TARGET_MISSING"])
    assert flags == []


def test_target_missing_no_target_no_flag() -> None:
    """Без target правило просто пропускается."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    flags = check_target_missing(df, None, {}, DEFAULT_THRESHOLDS["TARGET_MISSING"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 2. LEAKAGE_SUSPICION (critical)
# ──────────────────────────────────────────────────────────────────────────────


def test_leakage_high_correlation_flags() -> None:
    """Признак с |r| ≈ 1 даёт флаг по корреляционной ветке."""
    meta = {
        "target_correlation_by_column": {"feat_leak": 0.99, "feat_ok": 0.1},
        "target_mutual_information_by_column": {"feat_leak": 0.05, "feat_ok": 0.05},
    }
    flags = check_leakage_suspicion(pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["LEAKAGE"])
    assert len(flags) == 1
    assert flags[0].context["column"] == "feat_leak"
    assert flags[0].context["method"] == "pearson"


def test_leakage_high_mi_flags() -> None:
    """Признак с MI > 0.9 даёт флаг по MI-ветке (когда корреляция ниже порога)."""
    meta = {
        "target_correlation_by_column": {"feat_leak": 0.5},
        "target_mutual_information_by_column": {"feat_leak": 0.95},
    }
    flags = check_leakage_suspicion(pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["LEAKAGE"])
    assert len(flags) == 1
    assert flags[0].context["method"] == "mutual_information"


def test_leakage_no_target_returns_empty() -> None:
    flags = check_leakage_suspicion(pd.DataFrame(), None, {}, DEFAULT_THRESHOLDS["LEAKAGE"])
    assert flags == []


def test_leakage_clean_data_no_flag() -> None:
    meta = {
        "target_correlation_by_column": {"a": 0.2, "b": -0.3},
        "target_mutual_information_by_column": {"a": 0.05, "b": 0.04},
    }
    flags = check_leakage_suspicion(pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["LEAKAGE"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 3. HIGH_MISSING (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_high_missing_above_30_pct_flags_each_column() -> None:
    meta = {"missing_by_column": {"col_a": 0.4, "col_b": 0.5, "col_ok": 0.1}}
    flags = check_high_missing(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["HIGH_MISSING"])
    flagged = sorted(f.context["column"] for f in flags)
    assert flagged == ["col_a", "col_b"]


def test_high_missing_below_threshold_no_flag() -> None:
    meta = {"missing_by_column": {"col_a": 0.1, "col_b": 0.2}}
    flags = check_high_missing(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["HIGH_MISSING"])
    assert flags == []


def test_high_missing_skips_target_column() -> None:
    """target обрабатывается отдельным правилом TARGET_MISSING — здесь пропускается."""
    meta = {"missing_by_column": {"target": 0.7, "feature": 0.1}}
    flags = check_high_missing(pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["HIGH_MISSING"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 4. DUPLICATES (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_duplicates_above_threshold_flags() -> None:
    meta = {"duplicate_rows_pct": 0.10, "n_rows": 100}
    flags = check_duplicates(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["DUPLICATES"])
    assert len(flags) == 1
    assert flags[0].context["n_duplicates"] == 10


def test_duplicates_below_threshold_no_flag() -> None:
    meta = {"duplicate_rows_pct": 0.01, "n_rows": 100}
    flags = check_duplicates(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["DUPLICATES"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 5. IMBALANCE_BINARY (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_imbalance_binary_severe_flags() -> None:
    meta = {
        "target_kind": "categorical",
        "target_value_counts": {"0": 950, "1": 50},
        "target_imbalance_ratio": 19.0,
    }
    flags = check_imbalance_binary(
        pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["IMBALANCE_BINARY"]
    )
    assert len(flags) == 1


def test_imbalance_binary_balanced_no_flag() -> None:
    meta = {
        "target_kind": "categorical",
        "target_value_counts": {"0": 500, "1": 500},
        "target_imbalance_ratio": 1.0,
    }
    flags = check_imbalance_binary(
        pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["IMBALANCE_BINARY"]
    )
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 6. IMBALANCE_MULTICLASS (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_imbalance_multiclass_small_class_flags() -> None:
    meta = {
        "target_kind": "categorical",
        "target_value_counts": {"a": 500, "b": 400, "c": 10},
    }
    flags = check_imbalance_multiclass(
        pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["IMBALANCE_MULTICLASS"]
    )
    assert len(flags) == 1
    assert flags[0].context["min_class"] == "c"
    assert flags[0].context["min_class_size"] == 10


def test_imbalance_multiclass_all_classes_large_no_flag() -> None:
    meta = {
        "target_kind": "categorical",
        "target_value_counts": {"a": 200, "b": 200, "c": 100},
    }
    flags = check_imbalance_multiclass(
        pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["IMBALANCE_MULTICLASS"]
    )
    assert flags == []


def test_imbalance_multiclass_binary_skipped() -> None:
    """Бинарный target → правило не для него (его смотрит IMBALANCE_BINARY)."""
    meta = {"target_kind": "categorical", "target_value_counts": {"0": 5, "1": 5}}
    flags = check_imbalance_multiclass(
        pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["IMBALANCE_MULTICLASS"]
    )
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 7. SMALL_DATASET (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_small_dataset_flags() -> None:
    meta = {"sampling": {"original_size": 50}, "n_rows": 50}
    flags = check_small_dataset(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["SMALL_DATASET"])
    assert len(flags) == 1


def test_small_dataset_uses_original_size_when_sampled() -> None:
    """Если был сэмплинг, проверяем оригинальный размер, а не сэмпл."""
    meta = {"sampling": {"original_size": 200_000}, "n_rows": 50_000}
    flags = check_small_dataset(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["SMALL_DATASET"])
    assert flags == []


def test_small_dataset_above_threshold_no_flag() -> None:
    meta = {"sampling": {"original_size": 1000}, "n_rows": 1000}
    flags = check_small_dataset(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["SMALL_DATASET"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 8. TOO_FEW_FEATURES (warning)
# ──────────────────────────────────────────────────────────────────────────────


def test_too_few_features_with_target_flags() -> None:
    """3 колонки минус target = 2 признака, что меньше порога 3."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    meta = {"n_cols": 3}
    flags = check_too_few_features(df, "target", meta, DEFAULT_THRESHOLDS["TOO_FEW_FEATURES"])
    assert len(flags) == 1
    assert flags[0].context["n_features"] == 2


def test_too_few_features_enough_no_flag() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "target": [0]})
    meta = {"n_cols": 5}
    flags = check_too_few_features(df, "target", meta, DEFAULT_THRESHOLDS["TOO_FEW_FEATURES"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 9. LOW_VARIANCE (info)
# ──────────────────────────────────────────────────────────────────────────────


def test_low_variance_numeric_flags() -> None:
    meta = {
        "low_variance_numeric_cols": ["constant_col"],
        "low_variance_categorical_cols": [],
        "entropy_by_column": {},
    }
    flags = check_low_variance(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["LOW_VARIANCE"])
    assert len(flags) == 1
    assert flags[0].context["type"] == "numeric"


def test_low_variance_categorical_flags() -> None:
    meta = {
        "low_variance_numeric_cols": [],
        "low_variance_categorical_cols": ["dominant_cat"],
        "entropy_by_column": {"dominant_cat": 0.05},
    }
    flags = check_low_variance(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["LOW_VARIANCE"])
    assert len(flags) == 1
    assert flags[0].context["type"] == "categorical"


def test_low_variance_no_problem_no_flag() -> None:
    meta = {
        "low_variance_numeric_cols": [],
        "low_variance_categorical_cols": [],
        "entropy_by_column": {},
    }
    flags = check_low_variance(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["LOW_VARIANCE"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 10. HIGH_CARDINALITY (info)
# ──────────────────────────────────────────────────────────────────────────────


def test_high_cardinality_id_like_flags() -> None:
    meta = {
        "high_cardinality_cols": ["uuid_col"],
        "cardinality_by_column": {"uuid_col": 0.95},
    }
    flags = check_high_cardinality(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["HIGH_CARDINALITY"])
    assert len(flags) == 1
    assert flags[0].context["cardinality_ratio"] == 0.95


def test_high_cardinality_normal_categorical_no_flag() -> None:
    meta = {"high_cardinality_cols": [], "cardinality_by_column": {"sex": 0.001}}
    flags = check_high_cardinality(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["HIGH_CARDINALITY"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 11. OUTLIERS (info)
# ──────────────────────────────────────────────────────────────────────────────


def test_outliers_above_threshold_flags() -> None:
    meta = {"outliers_by_column": {"income": 0.10, "age": 0.01}}
    flags = check_outliers(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["OUTLIERS"])
    assert len(flags) == 1
    assert flags[0].context["column"] == "income"


def test_outliers_below_threshold_no_flag() -> None:
    meta = {"outliers_by_column": {"x": 0.01, "y": 0.005}}
    flags = check_outliers(pd.DataFrame(), None, meta, DEFAULT_THRESHOLDS["OUTLIERS"])
    assert flags == []


def test_outliers_target_excluded() -> None:
    """Выбросы в target — это другая природа, правило их игнорирует."""
    meta = {"outliers_by_column": {"target": 0.5}}
    flags = check_outliers(pd.DataFrame(), "target", meta, DEFAULT_THRESHOLDS["OUTLIERS"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 12. DATE_NOT_PARSED (info)
# ──────────────────────────────────────────────────────────────────────────────


def test_date_not_parsed_string_dates_flagged() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "created_at": pd.date_range("2024-01-01", periods=n)
            .strftime("%Y-%m-%d")
            .tolist(),
            "name": [chr(65 + i % 3) for i in range(n)],
        }
    )
    flags = check_date_not_parsed(df, None, {}, DEFAULT_THRESHOLDS["DATE_NOT_PARSED"])
    assert len(flags) == 1 and flags[0].context["column"] == "created_at"


def test_date_not_parsed_actual_datetime_skipped() -> None:
    """Уже распарсенный datetime не должен срабатывать (dtype не object)."""
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=10)})
    flags = check_date_not_parsed(df, None, {}, DEFAULT_THRESHOLDS["DATE_NOT_PARSED"])
    assert flags == []


def test_date_not_parsed_arbitrary_strings_skipped() -> None:
    df = pd.DataFrame({"name": ["foo", "bar", "baz"] * 30})
    flags = check_date_not_parsed(df, None, {}, DEFAULT_THRESHOLDS["DATE_NOT_PARSED"])
    assert flags == []


# ──────────────────────────────────────────────────────────────────────────────
# 13. Интеграция: run_quality_checks с реальной БД
# ──────────────────────────────────────────────────────────────────────────────


def test_run_quality_checks_clean_data_minimal_flags(db_session) -> None:
    """На чистых синтетических данных Iris-подобного датасета флагов мало (≤ 1)."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "sepal_length": rng.normal(5.8, 0.8, n),
            "sepal_width": rng.normal(3.0, 0.4, n),
            "petal_length": rng.normal(3.7, 1.7, n),
            "petal_width": rng.normal(1.2, 0.7, n),
            "species": rng.choice(["setosa", "versicolor", "virginica"], n),
        }
    )
    meta = compute_meta_features(df, target_col="species")
    flags = run_quality_checks(df, "species", meta, uuid.uuid4(), db_session)
    # На чистых данных серьёзных флагов быть не должно — допустимы 0..1.
    assert len(flags) <= 1


def test_run_quality_checks_dirty_data_multiple_flags(db_session) -> None:
    """На грязном синтетическом датасете срабатывают минимум 5 правил."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.choice([0, 1], n, p=[0.92, 0.08])
    df = pd.DataFrame(
        {
            "feat_id": [str(uuid.uuid4()) for _ in range(n)],  # high cardinality
            "feat_constant": np.zeros(n),  # low variance
            "feat_outliers": rng.normal(0, 1, n),
            "feat_missing": [
                v if i % 2 == 0 else None for i, v in enumerate(rng.normal(0, 1, n))
            ],
            "feat_leak": target.astype(float) + rng.normal(0, 0.001, n),
            "feat_date_str": pd.date_range("2024-01-01", periods=n, freq="D")
            .strftime("%Y-%m-%d")
            .tolist(),
            "target": target,
        }
    )
    df.loc[:20, "feat_outliers"] = 100.0  # выбросы
    df = pd.concat([df, df.iloc[:30]], ignore_index=True)  # дубликаты

    meta = compute_meta_features(df, target_col="target")
    flags = run_quality_checks(df, "target", meta, uuid.uuid4(), db_session)
    assert len(flags) >= 5
    # Должны сработать хотя бы по разу: leakage, missing, duplicates, imbalance.
    rule_codes_seen = set()
    # Связь rule_id → code пойдёт через сессию: rule_id из flag — UUID.
    # Сравниваем по message (точно содержит триггер) — этого достаточно для smoke.
    messages = " ".join(f.message for f in flags)
    assert "утеч" in messages.lower()
    assert "пропуск" in messages.lower()
    assert "дублик" in messages.lower()
    assert "дисбаланс" in messages.lower()
