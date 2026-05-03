"""
Unit-тесты для рекомендатера типа задачи (Слой 1 + Слой 2).

Покрытие согласно плану Phase 3 (минимум 14 тестов):
- 9 тестов на дерево правил Слоя 1 (по одному на каждый случай из
  recommender-rules.md, раздел «Тестирование правил»).
- 5 тестов на склейку слоёв и инфраструктуру (graceful degradation,
  hybrid, JSON-сериализация, стабильность хеша CANONICAL_FEATURE_ORDER).
- 2 e2e smoke на реальных Iris/Titanic-meta (опциональные, но дают
  уверенность что весь пайплайн работает на типичных данных).

Все тесты — чистые unit без БД и файловой системы (кроме e2e, которые
гоняют профайлер на in-memory DataFrame).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from app.services import task_recommender
from app.services.profiler import compute_meta_features
from app.services.task_recommender import (
    apply_rules,
    recommend_task,
    _load_meta_classifier_safe,
)
from ml.feature_vector import compute_feature_order_hash


# Зафиксированный SHA-256 от ','.join(CANONICAL_FEATURE_ORDER) на момент Phase 2.
# При изменении порядка фичей хеш изменится — тест ниже это поймает,
# напомнит о необходимости перезаливки каталога и переанализа всех датасетов.
EXPECTED_FEATURE_ORDER_HASH = (
    "a28f60492f82e9cd52d100e3cad98910f3746bcd2d60a73cc4399bb5ce4706c6"
)


@pytest.fixture(autouse=True)
def _reset_meta_classifier_cache():
    """
    Сбрасывает module-level кеш модели/scaler'а перед каждым тестом.

    Без этого test_load_meta_classifier_safe_returns_none_when_missing мог бы
    оставить _MODEL=None / _LOADED=True, и следующий тест (например, hybrid)
    не смог бы загрузить реальные артефакты.
    """
    task_recommender._MODEL = None
    task_recommender._SCALER = None
    task_recommender._LOADED = False
    yield
    task_recommender._MODEL = None
    task_recommender._SCALER = None
    task_recommender._LOADED = False


def _make_meta(**overrides: Any) -> dict[str, Any]:
    """
    Минимальный шаблон meta-features для тестов правил. Заполнен дефолтами,
    которые соответствуют «среднему здоровому» датасету; конкретный тест
    переопределяет нужные поля через kwargs.
    """
    base: dict[str, Any] = {
        "n_rows": 1000,
        "n_cols": 10,
        "memory_mb": 0.1,
        "total_missing_pct": 0.0,
        "max_col_missing_pct": 0.0,
        "duplicate_rows_pct": 0.0,
        "mean_skewness": 0.0,
        "mean_kurtosis": 0.0,
        "normality_test_pvalue": 0.5,
        "outliers_pct": 0.01,
        "max_abs_correlation": 0.5,
        "target_correlation_max": 0.3,
        "target_mutual_information_max": 0.2,
        "target_mutual_information_mean": 0.1,
        # target-specific (по умолчанию None — кластеризация)
        "target_kind": None,
        "target_n_unique": None,
        "target_imbalance_ratio": None,
        "target_class_entropy": None,
        "target_skewness": None,
        "target_value_counts": None,
        "target_column": None,
    }
    base.update(overrides)
    return base


# =============================================================================
#                  9 ТЕСТОВ НА ПРАВИЛА СЛОЯ 1 (ВЕТКИ 1, 3, 4)
# =============================================================================


def test_numeric_binary_target():
    """Ветка 3, NUMERIC_BINARY_TARGET: target числовой 0/1 → BINARY."""
    meta = _make_meta(target_kind="regression", target_n_unique=2)
    result = apply_rules(meta, target_column="label")
    assert result.task_type_code == "BINARY_CLASSIFICATION"
    assert result.confidence == pytest.approx(0.95)
    assert result.requires_ml is False
    assert "NUMERIC_BINARY_TARGET" in [r.code for r in result.applied_rules]


def test_numeric_many_values():
    """Ветка 3, NUMERIC_TARGET_MANY_VALUES: numeric с >20 значениями → REGRESSION."""
    meta = _make_meta(target_kind="regression", target_n_unique=200)
    result = apply_rules(meta, target_column="price")
    assert result.task_type_code == "REGRESSION"
    assert result.confidence == pytest.approx(0.95)
    assert result.requires_ml is False
    assert "NUMERIC_TARGET_MANY_VALUES" in [r.code for r in result.applied_rules]


def test_categorical_2_classes():
    """Ветка 4, BINARY_BALANCED: категориальный target, 2 класса, баланс ≤ 10:1."""
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=2,
        target_imbalance_ratio=2.0,
        target_value_counts={"yes": 600, "no": 400},
    )
    result = apply_rules(meta, target_column="label")
    assert result.task_type_code == "BINARY_CLASSIFICATION"
    assert result.confidence == pytest.approx(0.95)
    assert result.requires_ml is False
    codes = [r.code for r in result.applied_rules]
    assert "BINARY_BALANCED" in codes


def test_categorical_5_classes():
    """Ветка 4, MULTICLASS: 5 классов с min_class_size ≥ 50 → MULTICLASS conf=0.95."""
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=5,
        target_value_counts={
            "a": 200, "b": 200, "c": 200, "d": 200, "e": 200,
        },
    )
    result = apply_rules(meta, target_column="cls")
    assert result.task_type_code == "MULTICLASS_CLASSIFICATION"
    assert result.confidence == pytest.approx(0.95)
    assert result.requires_ml is False
    assert "MULTICLASS" in [r.code for r in result.applied_rules]


def test_no_target_low_dim():
    """Ветка 1, NO_TARGET_LOW_DIM: target=None, n_cols ≤ 10 → CLUSTERING conf=0.9."""
    meta = _make_meta(n_cols=5)  # target_kind=None, target_column=None по дефолту
    result = apply_rules(meta, target_column=None)
    assert result.task_type_code == "CLUSTERING"
    assert result.confidence == pytest.approx(0.9)
    assert result.requires_ml is False
    assert "NO_TARGET_LOW_DIM" in [r.code for r in result.applied_rules]


def test_no_target_high_dim():
    """Ветка 1, NO_TARGET_HIGH_DIM: target=None, n_cols > 10 → CLUSTERING + DIMENSIONALITY_REDUCTION."""
    meta = _make_meta(n_cols=50)
    result = apply_rules(meta, target_column=None)
    assert result.task_type_code == "CLUSTERING"
    assert result.confidence == pytest.approx(0.8)
    assert result.requires_ml is False
    codes = [r.code for r in result.applied_rules]
    # Оба правила фигурируют как метки в applied_rules; финальный task_type — CLUSTERING.
    assert "NO_TARGET_HIGH_DIM" in codes
    assert "DIMENSIONALITY_REDUCTION" in codes


def test_ambiguous_numeric_delegates_to_ml():
    """Ветка 3, AMBIGUOUS_NUMERIC_TARGET: numeric target, 5 уникальных → requires_ml=True."""
    meta = _make_meta(target_kind="regression", target_n_unique=5)
    result = apply_rules(meta, target_column="rating")
    assert result.requires_ml is True
    assert result.confidence == pytest.approx(0.5)
    assert "AMBIGUOUS_NUMERIC_TARGET" in [r.code for r in result.applied_rules]


def test_target_looks_like_id():
    """Ветка 4, TARGET_LOOKS_LIKE_ID: categorical, n_classes > 20, cardinality_ratio > 0.5 → NOT_READY."""
    # 600 уникальных при 1000 строк → cardinality_ratio = 0.6 > 0.5
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=600,
        n_rows=1000,
        # value_counts не критичен — мы попадаем в ветку n_classes > 20.
        target_value_counts={f"v{i}": 1 for i in range(600)},
    )
    result = apply_rules(meta, target_column="user_id")
    assert result.task_type_code == "NOT_READY"
    assert result.confidence == pytest.approx(0.8)
    assert result.requires_ml is False
    assert "TARGET_LOOKS_LIKE_ID" in [r.code for r in result.applied_rules]


def test_only_one_class():
    """Ветка 4, ONLY_ONE_CLASS: categorical с n_classes < 2 → NOT_READY."""
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=1,
        target_value_counts={"only": 1000},
    )
    result = apply_rules(meta, target_column="label")
    assert result.task_type_code == "NOT_READY"
    assert result.confidence == pytest.approx(0.99)
    assert result.requires_ml is False
    assert "ONLY_ONE_CLASS" in [r.code for r in result.applied_rules]


# =============================================================================
#                  5 ТЕСТОВ НА СКЛЕЙКУ И ИНФРАСТРУКТУРУ
# =============================================================================


def test_recommend_with_rules_only_when_model_missing(monkeypatch):
    """
    При отсутствии модели и scaler'а recommend_task должен работать на чистых
    правилах: source="rules", в explanation — пометка про недоступность модели.
    """
    monkeypatch.setattr(
        task_recommender,
        "_load_meta_classifier_safe",
        lambda *args, **kwargs: (None, None),
    )
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=2,
        target_imbalance_ratio=2.0,
        target_value_counts={"yes": 600, "no": 400},
    )
    rec = recommend_task(meta, target_column="label")
    assert rec.source == "rules"
    assert rec.task_type_code == "BINARY_CLASSIFICATION"
    assert rec.ml_probabilities is None
    assert "только правила" in rec.explanation


def test_recommend_hybrid_for_borderline_case():
    """
    Пограничный случай (numeric target, 5 значений) с реально загруженной
    моделью: source должен стать "ml" или "hybrid" (не "rules").
    """
    model, scaler = _load_meta_classifier_safe()
    if model is None or scaler is None:
        pytest.skip(
            "Meta-classifier not trained yet (run `make train-meta`). "
            "Этот тест проверяет именно режим с моделью."
        )
    meta = _make_meta(target_kind="regression", target_n_unique=5)
    rec = recommend_task(meta, target_column="rating")
    assert rec.source in {"ml", "hybrid"}
    assert rec.ml_probabilities is not None
    assert sum(rec.ml_probabilities.values()) == pytest.approx(1.0, abs=0.01)


def test_task_recommendation_serializable_to_json():
    """
    `TaskRecommendation.model_dump()` должен давать JSON-совместимый dict
    для записи в analysis_results.task_recommendation JSONB (Phase 6).
    """
    meta = _make_meta(
        target_kind="categorical",
        target_n_unique=3,
        target_value_counts={"a": 100, "b": 100, "c": 100},
    )
    rec = recommend_task(meta, target_column="cls")
    payload = rec.model_dump()
    # Если dump'ом можно сериализовать через json — значит JSONB примет.
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["task_type_code"] == rec.task_type_code
    assert decoded["source"] == rec.source
    assert isinstance(decoded["applied_rules"], list)


def test_canonical_feature_order_hash_stable():
    """
    Хеш CANONICAL_FEATURE_ORDER должен быть стабильным между запусками.

    При изменении порядка фичей в Phase 2 этот тест громко упадёт.
    Чтобы его починить: (1) обнови EXPECTED_FEATURE_ORDER_HASH здесь,
    (2) ОБЯЗАТЕЛЬНО `make seed-catalog` (Phase 4) и переанализ всех существующих
    датасетов — иначе старые embedding'и в БД станут несовместимыми с новыми.
    """
    assert compute_feature_order_hash() == EXPECTED_FEATURE_ORDER_HASH, (
        "CANONICAL_FEATURE_ORDER изменился! Embedding'и пользовательских "
        "датасетов и каталога стали несовместимы. Перезалей каталог через "
        "`make seed-catalog` и обнови EXPECTED_FEATURE_ORDER_HASH в этом тесте."
    )


def test_load_meta_classifier_safe_returns_none_when_missing(monkeypatch, tmp_path):
    """
    При отсутствии файлов модели/scaler'а функция должна вернуть (None, None)
    без исключения.
    """
    # Перенаправляем пути в пустую временную папку.
    monkeypatch.setattr(
        task_recommender, "ML_MODELS_DIR", tmp_path
    )
    monkeypatch.setattr(
        task_recommender, "META_CLASSIFIER_PATH", tmp_path / "meta_classifier.pkl"
    )
    monkeypatch.setattr(
        task_recommender, "SCALER_PATH", tmp_path / "scaler.pkl"
    )
    model, scaler = _load_meta_classifier_safe(force_reload=True)
    assert model is None
    assert scaler is None


# =============================================================================
#                  E2E SMOKE: ПОЛНЫЙ ПАЙПЛАЙН НА РЕАЛЬНЫХ MET-A
# =============================================================================


def test_recommend_iris_returns_multiclass():
    """
    E2E smoke: meta из реального Iris (3 класса) → MULTICLASS.

    Не зависит от наличия обученной модели: даже если модель есть, правило
    BINARY_BALANCED/MULTICLASS даст высокую confidence и Слой 2 не вызовется.
    """
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame.copy()
    meta = compute_meta_features(df, target_col="target")
    rec = recommend_task(meta, target_column="target")
    assert rec.task_type_code == "MULTICLASS_CLASSIFICATION"
    assert rec.confidence >= 0.8


def test_recommend_no_target_returns_clustering():
    """
    E2E smoke: meta без target → CLUSTERING. Это стабильно работает даже без
    обученной модели (Ветка 1 правил).
    """
    iris = load_iris(as_frame=True)
    df: pd.DataFrame = iris.frame.copy().drop(columns=["target"])
    meta = compute_meta_features(df, target_col=None)
    rec = recommend_task(meta, target_column=None)
    assert rec.task_type_code == "CLUSTERING"
