"""
Обучение мета-классификатора Слоя 2 (Random Forest на meta-features).

Скрипт читает `real_set.json` + `synthetic_set.json`, отфильтровывает
CLUSTERING-записи (Слой 2 видит только 3 класса: BINARY/MULTICLASS/REGRESSION),
векторизует meta-features через `meta_features_to_vector`, обучает
StandardScaler и RandomForestClassifier, считает CV-метрики через
StratifiedKFold(5) и сохраняет:

- `backend/ml/models/scaler.pkl` — обученный StandardScaler
- `backend/ml/models/meta_classifier.pkl` — обученный RandomForestClassifier
- `backend/ml/models/meta_classifier_report.json` — CV-сводка для РПЗ

Гиперпараметры RF зафиксированы (без GridSearchCV) — на ~180 примерах
поиск по сетке не имеет смысла, см. `recommender-ml.md`.

Запуск: `make train-meta` (см. Makefile).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from ml.feature_vector import CANONICAL_FEATURE_ORDER, meta_features_to_vector


logger = logging.getLogger("train_meta_classifier")


# Пути к артефактам.
REAL_SET_PATH = Path("ml/data/real_set.json")
SYNTHETIC_SET_PATH = Path("ml/data/synthetic_set.json")
MODELS_DIR = Path("ml/models")
SCALER_PATH = MODELS_DIR / "scaler.pkl"
CLASSIFIER_PATH = MODELS_DIR / "meta_classifier.pkl"
REPORT_PATH = MODELS_DIR / "meta_classifier_report.json"


# Классы, которые видит Слой 2. CLUSTERING исключён — это задача Слоя 1.
TARGET_CLASSES = {
    "BINARY_CLASSIFICATION",
    "MULTICLASS_CLASSIFICATION",
    "REGRESSION",
}


# Гиперпараметры — фиксированы. См. `.knowledge/methods/recommender-ml.md`.
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}


# Архивная запись для РПЗ: итерации feature engineering на этой фазе.
# Шаг 1 (отвергнут) — 18 фичей с косвенным leakage через None-семантику.
# Шаг 2 (недостаточно) — 15 фичей, нет сигнала для BINARY vs MULTICLASS.
# Шаг 3 (финал) — 16 фичей с добавленной target_n_unique.
REMOVED_FEATURES_DUE_TO_STRUCTURAL_LEAKAGE = {
    "features": [
        "target_imbalance_ratio",
        "target_class_entropy",
        "target_skewness",
    ],
    "reason": (
        "Профайлер возвращает None для этих признаков на регрессионных датасетах "
        "(где target_kind == 'regression') и числовое значение на классификационных. "
        "Замена None → 0.0 в meta_features_to_vector превращала их в бинарный "
        "индикатор target_kind == 'categorical' — косвенный leakage, эквивалентный "
        "включению самого target_kind как фичи. Удаление снижает CV f1_macro с "
        "искусственного 0.987 до содержательного диапазона 0.85–0.92."
    ),
}

ADDED_FEATURES = {
    "features": ["target_n_unique"],
    "reason": (
        "Continuous descriptive statistic, computed via series.nunique() "
        "unconditionally (без зависимости от target_kind, не None). "
        "Canonical simple meta-feature per Vanschoren 2018 §3.2 "
        "('number of classes / target values'). Возвращает 2 для binary, "
        "3-20 для multiclass, сотни-тысячи для regression. Добавлен на третьей "
        "итерации Phase 2: после удаления target_imbalance_ratio/_class_entropy/_skewness "
        "Слой 2 потерял сигнал для различения BINARY vs MULTICLASS (CV f1_macro = 0.796); "
        "target_n_unique восстанавливает этот сигнал без leakage."
    ),
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _build_xy(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Собирает матрицу X (n_samples × n_features) и метки y из списка записей.

    Возвращает (X, y, source_counts), где source_counts — словарь
    {real: N, synthetic: M} для отчёта.
    """
    rows = []
    labels = []
    source_counts = {"real": 0, "synthetic": 0}
    for rec in records:
        if rec["task_type_code"] not in TARGET_CLASSES:
            continue
        meta = rec["meta_features"]
        rows.append(meta_features_to_vector(meta))
        labels.append(rec["task_type_code"])
        if rec.get("source") == "synthetic":
            source_counts["synthetic"] += 1
        else:
            source_counts["real"] += 1
    X = np.vstack(rows) if rows else np.empty((0, len(CANONICAL_FEATURE_ORDER)))
    y = np.array(labels)
    return X, y, source_counts


def _cross_validate(
    model: RandomForestClassifier,
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    StratifiedKFold(5) cross-validation.
    Метрики: accuracy, f1_macro, f1_weighted (mean ± std по фолдам).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X_scaled,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "accuracy": {
            "mean": float(cv_results["test_accuracy"].mean()),
            "std": float(cv_results["test_accuracy"].std()),
        },
        "f1_macro": {
            "mean": float(cv_results["test_f1_macro"].mean()),
            "std": float(cv_results["test_f1_macro"].std()),
        },
        "f1_weighted": {
            "mean": float(cv_results["test_f1_weighted"].mean()),
            "std": float(cv_results["test_f1_weighted"].std()),
        },
    }


def _confusion_matrix_cv(
    model: RandomForestClassifier,
    X_scaled: np.ndarray,
    y: np.ndarray,
    classes: list[str],
) -> list[list[int]]:
    """
    Считает confusion matrix по out-of-fold предсказаниям всех 5 фолдов
    StratifiedKFold (так покрытие — все 100% обучающей выборки).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.empty_like(y)
    for train_idx, test_idx in cv.split(X_scaled, y):
        clone_model = RandomForestClassifier(**RF_PARAMS)
        clone_model.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = clone_model.predict(X_scaled[test_idx])
    cm = confusion_matrix(y, y_pred, labels=classes)
    return cm.tolist()


def train_meta_classifier() -> dict[str, Any]:
    """Главная точка входа. Возвращает финальный отчёт (тоже пишется в JSON)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    real = _load_records(REAL_SET_PATH)
    synthetic = _load_records(SYNTHETIC_SET_PATH)
    n_clustering_excluded = sum(
        1 for r in real if r["task_type_code"] == "CLUSTERING"
    )
    all_records = real + synthetic

    X, y, source_counts = _build_xy(all_records)
    if len(X) == 0:
        raise RuntimeError(
            "Empty training set. Run `make build-real-set && make build-synthetic-set` first."
        )

    logger.info(
        "Training set: %d total (real=%d, synthetic=%d, %d clustering excluded)",
        len(X), source_counts["real"], source_counts["synthetic"],
        n_clustering_excluded,
    )

    # 1. StandardScaler — обучается на всей выборке, чтобы потом тот же
    # scaler применять и к пользовательским датасетам в Phase 4.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Saved scaler → %s", SCALER_PATH)

    # 2. CV-оценка качества на отмасштабированных признаках.
    cv_metrics = _cross_validate(RandomForestClassifier(**RF_PARAMS), X_scaled, y)
    logger.info(
        "CV f1_macro = %.4f ± %.4f, accuracy = %.4f ± %.4f",
        cv_metrics["f1_macro"]["mean"], cv_metrics["f1_macro"]["std"],
        cv_metrics["accuracy"]["mean"], cv_metrics["accuracy"]["std"],
    )

    # 3. Confusion matrix через OOF-предсказания.
    classes = sorted(set(y))
    cm = _confusion_matrix_cv(RandomForestClassifier(**RF_PARAMS), X_scaled, y, classes)

    # 4. Финальное обучение на всей выборке.
    final_model = RandomForestClassifier(**RF_PARAMS)
    final_model.fit(X_scaled, y)
    joblib.dump(final_model, CLASSIFIER_PATH)
    logger.info("Saved classifier → %s", CLASSIFIER_PATH)

    # 5. Feature importance — top-10.
    importance_pairs = sorted(
        zip(CANONICAL_FEATURE_ORDER, final_model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top10 = {name: float(score) for name, score in importance_pairs[:10]}

    class_distribution = {cls: int((y == cls).sum()) for cls in classes}

    report = {
        "n_total": int(len(X)),
        "n_real": source_counts["real"],
        "n_synthetic": source_counts["synthetic"],
        "n_clustering_excluded": n_clustering_excluded,
        "classes": classes,
        "class_distribution": class_distribution,
        "feature_order": CANONICAL_FEATURE_ORDER,
        "removed_features_due_to_structural_leakage": REMOVED_FEATURES_DUE_TO_STRUCTURAL_LEAKAGE,
        "added_features": ADDED_FEATURES,
        "cv_metrics": cv_metrics,
        "confusion_matrix": cm,
        "feature_importances_top10": top10,
        "hyperparameters": RF_PARAMS,
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved report → %s", REPORT_PATH)

    return report


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    train_meta_classifier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
