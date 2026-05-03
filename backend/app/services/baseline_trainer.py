"""
Сервис обучения baseline-моделей: 2 модели на задачу + кросс-валидация.

Модуль реализует научную часть Спринта 3 — обучение «эталонных» моделей
для каждого типа ML-задачи. Цель — дать пользователю отправную точку
качества: «вот что можно получить простыми методами на ваших данных,
ваша сложная модель должна это побеждать».

Алгоритм для классификации (BINARY / MULTICLASS):
- LogisticRegression(max_iter=200, class_weight="balanced") — линейная нижняя граница.
- RandomForestClassifier(n_estimators=50, max_depth=5) — ансамблевый baseline.
- StratifiedKFold(5) — кросс-валидация со стратификацией по классам.
- Метрики: accuracy, precision, recall, f1, roc_auc (binary) /
  accuracy, f1_macro, f1_weighted (multiclass).

Алгоритм для регрессии:
- Ridge — линейная нижняя граница.
- RandomForestRegressor(n_estimators=50, max_depth=5) — ансамбль.
- KFold(5).
- Метрики: mae, rmse, r2 (rmse через sqrt от per-fold neg_mean_squared_error).

Для CLUSTERING / NOT_READY возвращается заглушка — задачи без целевой
переменной не имеют однозначного эталонного качества (см.
.knowledge/methods/baseline-training.md, раздел «Для кластеризации»).

Принципы реализации (см. CLAUDE.md и baseline-training.md):
- Превентивный лимит на время: стратифицированный сэмпл до 5000 строк +
  фиксированные гиперпараметры моделей. Никаких multiprocessing/signal/timeout —
  они несовместимы с asyncio event loop FastAPI и плодят зомби-процессы.
- Защита от утечки: колонки с флагом LEAKAGE_SUSPICION исключаются перед
  обучением (см. .knowledge/methods/quality-checks.md, правило 9).
- Кросс-валидация (5 фолдов) — основная защита от переобучения; в результате
  возвращается mean ± std по фолдам.

Источник: Bishop C. "Pattern Recognition and Machine Learning", Springer, 2006.
См. .knowledge/methods/baseline-training.md.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.analysis_result import AnalysisResult
from app.models.quality_flag import QualityFlag
from app.models.quality_rule import QualityRule
from app.services.dataset_service import read_dataset_full

logger = logging.getLogger(__name__)


# Превентивный лимит на размер выборки. На 5000 строках любая из моделей
# с фиксированными гиперпараметрами укладывается в 5-15 секунд независимо
# от исходного размера датасета (см. baseline-training.md, «Производительность»).
MAX_SAMPLE_SIZE = 5000

# Все рандомизированные операции (сэмпл, KFold, RF) — с фиксированным seed
# ради воспроизводимости (требование ТЗ).
RANDOM_STATE = 42

# Порог cardinality_ratio для решения «one-hot или дроп»:
# колонки с долей уникальных значений < 0.1 — кодируются в dummies;
# >= 0.1 — отбрасываются (Name/Ticket/ID-like раздули бы матрицу X).
ONE_HOT_CARDINALITY_THRESHOLD = 0.1

# Сколько топовых признаков по importance оставить в результате.
TOP_FEATURES_COUNT = 10


# =============================================================================
#                       1. ПРЕПРОЦЕССИНГ
# =============================================================================


def _preprocess(
    df: pd.DataFrame,
    target_col: str,
    leakage_cols: list[str],
    meta: dict[str, Any],
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Приводит сырой DataFrame к виду (X, y), пригодному для обучения sklearn.

    Шаги:
    1. Удаление строк с NaN в target_col (target обязателен для supervised).
    2. Сэмпл до MAX_SAMPLE_SIZE строк:
       - классификация — train_test_split со stratify=y (сохраняет пропорции
         классов даже на сильном дисбалансе 1:1000);
       - регрессия — random sample.
       Если строк меньше MAX_SAMPLE_SIZE — сэмпл не выполняется.
    3. Отделение y, удаление target_col + leakage_cols + datetime-колонок +
       высококардинальных категориальных (>= ONE_HOT_CARDINALITY_THRESHOLD) из X.
    4. Имьют пропусков: медиана для числовых, мода для категориальных.
    5. One-hot encoding для оставшихся категориальных
       (cardinality_ratio < ONE_HOT_CARDINALITY_THRESHOLD).
    6. StandardScaler на исходные числовые признаки (one-hot не масштабируется —
       0/1-колонки и так в одном масштабе).

    Особый кейс: для MULTICLASS_CLASSIFICATION с числовым target значения
    приводятся к категориальным кодам через `pd.Categorical(...).codes` —
    sklearn-классификаторы примут [0.0, 1.0, 2.0] как 3 класса, но scoring
    f1_macro на float-метках работает менее надёжно.

    Args:
        df: исходный DataFrame.
        target_col: имя целевого столбца.
        leakage_cols: колонки с подозрением на утечку (исключаются).
        meta: meta-features из profiler.compute_meta_features (нужен ключ
            cardinality_by_column для выбора кодирования категориальных).
        task_type: тип задачи — определяет тип сэмплирования.

    Returns:
        Кортеж (X, y) — числовая матрица признаков и целевая переменная.
    """
    # 1. Удаляем строки с NaN в target.
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

    # 2. Сэмпл до MAX_SAMPLE_SIZE — гарантия укладывания в 5-15 секунд.
    if len(df_clean) > MAX_SAMPLE_SIZE:
        if task_type in ("BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"):
            df_clean, _ = train_test_split(
                df_clean,
                train_size=MAX_SAMPLE_SIZE,
                stratify=df_clean[target_col],
                random_state=RANDOM_STATE,
            )
            df_clean = df_clean.reset_index(drop=True)
        else:  # REGRESSION
            df_clean = df_clean.sample(
                n=MAX_SAMPLE_SIZE, random_state=RANDOM_STATE
            ).reset_index(drop=True)

    # 3. Отделяем y и формируем X без leakage/target/datetime/high-card.
    y = df_clean[target_col].copy()

    cardinality_by_col = meta.get("cardinality_by_column") or {}

    cols_to_drop: set[str] = {target_col, *leakage_cols}
    # Datetime-колонки sklearn не принимает — отбрасываем явно.
    for col in df_clean.columns:
        if df_clean[col].dtype.kind in {"M", "m"}:
            cols_to_drop.add(col)
    # Высококардинальные категориальные (>= ONE_HOT_CARDINALITY_THRESHOLD)
    # исключаем — иначе один Name/Ticket породил бы сотни dummy-колонок.
    high_card_cols = [
        col
        for col, ratio in cardinality_by_col.items()
        if float(ratio) >= ONE_HOT_CARDINALITY_THRESHOLD
    ]
    cols_to_drop.update(high_card_cols)

    X = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

    # 4. Имьют пропусков: численные → медиана, категориальные → мода.
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in {"i", "u", "f"}]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    for col in numeric_cols:
        median_val = X[col].median()
        # Если вся колонка — NaN, медиана тоже NaN; подменяем нулём.
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)

    for col in categorical_cols:
        mode_series = X[col].mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else "MISSING"
        X[col] = X[col].fillna(fill_val)

    # 5. One-hot для оставшихся категориальных. После шага 3 в X остались
    # только колонки с ratio < ONE_HOT_CARDINALITY_THRESHOLD — кодируем все.
    if categorical_cols:
        X = pd.get_dummies(
            X, columns=categorical_cols, drop_first=False, dummy_na=False
        )

    # Имена колонок приводим к строкам — иначе после get_dummies могут
    # появиться имена-числа (если оригинальная категория была числовой),
    # и они плохо сериализуются в JSONB.
    X.columns = [str(c) for c in X.columns]

    # 6. StandardScaler — только на исходные числовые. Бинарные one-hot
    # колонки оставляем как 0/1: масштабировать их не имеет смысла, а
    # для LogReg/Ridge это не критично (гладкий градиент по 0/1 работает).
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Особый кейс: MULTICLASS с числовым target. sklearn-классификатор примет
    # [0.0, 1.0, 2.0] как 3 класса, но pd.Categorical.codes даёт чистые int —
    # f1_macro/f1_weighted на них считаются стабильнее.
    if task_type == "MULTICLASS_CLASSIFICATION" and pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=y.index)

    return X, y


# =============================================================================
#                       2. КРОСС-ВАЛИДАЦИЯ И МЕТРИКИ
# =============================================================================


def _scoring_for(task_type: str) -> list[str]:
    """Список scoring-имён для cross_validate в формате sklearn."""
    if task_type == "BINARY_CLASSIFICATION":
        return ["accuracy", "precision", "recall", "f1", "roc_auc"]
    if task_type == "MULTICLASS_CLASSIFICATION":
        # roc_auc для multiclass требует явного multi_class-параметра;
        # f1_macro/f1_weighted — стандартный набор для несбалансированных классов.
        return ["accuracy", "f1_macro", "f1_weighted"]
    if task_type == "REGRESSION":
        return ["neg_mean_absolute_error", "neg_mean_squared_error", "r2"]
    return []


def _aggregate_cv_results(
    cv_results: dict[str, np.ndarray], task_type: str
) -> dict[str, dict[str, float]]:
    """
    Преобразует test_<scoring>-массивы из cross_validate в человеческие метрики.

    Для регрессии:
    - neg_mean_absolute_error → mae (знак инвертируется);
    - neg_mean_squared_error → rmse (sqrt от каждого фолда, потом mean/std);
    - r2 без преобразования.
    """
    metrics: dict[str, dict[str, float]] = {}

    if task_type == "REGRESSION":
        neg_mae = cv_results["test_neg_mean_absolute_error"]
        mae_per_fold = -neg_mae
        metrics["mae"] = {
            "mean": float(mae_per_fold.mean()),
            "std": float(mae_per_fold.std()),
        }

        # RMSE считаем как sqrt(MSE_per_fold), а потом mean/std — это даёт
        # более интерпретируемое std, чем sqrt от усреднённого MSE.
        neg_mse = cv_results["test_neg_mean_squared_error"]
        rmse_per_fold = np.sqrt(-neg_mse)
        metrics["rmse"] = {
            "mean": float(rmse_per_fold.mean()),
            "std": float(rmse_per_fold.std()),
        }

        r2 = cv_results["test_r2"]
        metrics["r2"] = {"mean": float(r2.mean()), "std": float(r2.std())}
        return metrics

    for scoring in _scoring_for(task_type):
        arr = cv_results[f"test_{scoring}"]
        metrics[scoring] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return metrics


# =============================================================================
#                       3. КАТАЛОГ МОДЕЛЕЙ
# =============================================================================


def _make_classifier_models() -> dict[str, Any]:
    """
    Linear + ensemble baseline для классификации.

    Гиперпараметры зафиксированы (см. baseline-training.md, «Производительность»):
    LogReg — max_iter=200, class_weight="balanced".
    RF — n_estimators=50, max_depth=5, n_jobs=-1.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _make_regressor_models() -> dict[str, Any]:
    """Linear + ensemble baseline для регрессии."""
    return {
        "ridge": Ridge(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _utc_now_iso() -> str:
    """ISO-формат с суффиксом Z — для отображения в UI и записи в JSONB."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clustering_stub() -> dict[str, Any]:
    """
    Заглушка для CLUSTERING / NOT_READY.

    Baseline для кластеризации в Спринте 3 не делается: без знания целевых
    кластеров нет однозначного эталонного качества. UI показывает рекомендуемые
    алгоритмы (KMeans, DBSCAN) и метрики (Silhouette) без обучения.
    """
    return {
        "models": [],
        "metrics": {},
        "feature_importance": {},
        "excluded_columns_due_to_leakage": [],
        "n_rows_used": 0,
        "n_features_used": 0,
        "note": (
            "Baseline для этого типа задачи не реализован. "
            "Рекомендуемые алгоритмы: KMeans, DBSCAN."
        ),
        "trained_at": _utc_now_iso(),
    }


# =============================================================================
#                       4. ОСНОВНЫЕ ТОЧКИ ВХОДА
# =============================================================================


def train_baseline_from_df(
    df: pd.DataFrame,
    meta: dict[str, Any],
    leakage_cols: list[str],
    target_col: str,
    task_type: str,
) -> dict[str, Any]:
    """
    Обучает 2 baseline-модели по типу задачи и собирает результат-словарь.

    Точка входа без БД — её бьют unit-тесты, она же используется обёрткой
    `train_baseline()` после загрузки данных.

    Алгоритм:
    1. Если task_type ∈ {CLUSTERING, NOT_READY} — возвращаем заглушку.
    2. Препроцессинг через `_preprocess` (сэмпл, импутация, кодирование, scaling).
    3. Кросс-валидация (5 фолдов) для каждой из 2 моделей.
    4. Финальное обучение RandomForest на всём X для feature_importance топ-10.
    5. Сборка результата по контракту baseline-training.md.

    Args:
        df: сырой DataFrame.
        meta: meta-features из profiler (нужны для cardinality_by_column).
        leakage_cols: колонки, помеченные правилом LEAKAGE_SUSPICION.
        target_col: имя целевого столбца.
        task_type: код типа задачи (BINARY_CLASSIFICATION / MULTICLASS_CLASSIFICATION /
            REGRESSION / CLUSTERING / NOT_READY).

    Returns:
        Словарь по контракту baseline-training.md с ключами models, metrics,
        feature_importance, excluded_columns_due_to_leakage, n_rows_used,
        n_features_used, trained_at.

    Raises:
        NotImplementedError: для неизвестных task_type.
    """
    if task_type in ("CLUSTERING", "NOT_READY"):
        return _clustering_stub()

    if task_type in ("BINARY_CLASSIFICATION", "MULTICLASS_CLASSIFICATION"):
        models = _make_classifier_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    elif task_type == "REGRESSION":
        models = _make_regressor_models()
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        raise NotImplementedError(f"Baseline для {task_type!r} не поддерживается")

    X, y = _preprocess(df, target_col, leakage_cols, meta, task_type)
    scoring = _scoring_for(task_type)

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        metrics[name] = _aggregate_cv_results(cv_results, task_type)

    # Feature importance — обучаем RF на всём X. cross_validate не возвращает
    # обученные модели (return_estimator=False по умолчанию), и получить
    # importance из CV нельзя — нужно отдельное обучение.
    rf_model = models["random_forest"]
    rf_model.fit(X, y)
    importance_pairs = sorted(
        zip(X.columns, rf_model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )[:TOP_FEATURES_COUNT]
    feature_importance = {str(col): float(imp) for col, imp in importance_pairs}

    return {
        "models": list(models.keys()),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "excluded_columns_due_to_leakage": list(leakage_cols),
        "n_rows_used": int(X.shape[0]),
        "n_features_used": int(X.shape[1]),
        "trained_at": _utc_now_iso(),
    }


def _get_leakage_columns(db: Session, analysis_id: uuid.UUID) -> list[str]:
    """
    Достаёт имена колонок-подозреваемых в утечке из quality_flags.

    Источник — записи QualityFlag, чьё правило имеет code='LEAKAGE_SUSPICION';
    имя колонки лежит в context['column'] (см. quality_checker.py,
    `check_leakage_suspicion`).
    """
    stmt = (
        select(QualityFlag.context)
        .join(QualityRule, QualityFlag.rule_id == QualityRule.id)
        .where(QualityFlag.analysis_id == analysis_id)
        .where(QualityRule.code == "LEAKAGE_SUSPICION")
    )
    contexts = db.execute(stmt).scalars().all()
    cols: list[str] = []
    for ctx in contexts:
        if isinstance(ctx, dict):
            col = ctx.get("column")
            if isinstance(col, str):
                cols.append(col)
    return cols


def train_baseline(
    analysis_id: uuid.UUID,
    target_col: str,
    task_type: str,
    db: Session,
) -> dict[str, Any]:
    """
    Точка входа обучения baseline для пользовательского анализа.

    Функция синхронная и CPU-bound. Асинхронная обёртка делается на уровне
    оркестратора через `asyncio.to_thread()` (Phase 6, baseline_orchestrator.py) —
    она освобождает event loop, позволяя продолжать обслуживать другие HTTP-запросы.

    Алгоритм:
    1. Загрузить датасет, привязанный к анализу, целиком (как в analysis_service).
    2. Прочитать meta_features из analysis_results и leakage-колонки из quality_flags.
    3. Делегировать `train_baseline_from_df`.

    Никаких multiprocessing/signal/timeout — превентивные лимиты на размер
    выборки и гиперпараметры моделей гарантируют 5-15 секунд. При сбое
    (битый CSV, OOM, неизвестный task_type) — поднимается исключение,
    оркестратор переводит baseline_status в 'failed' с обрезанным error.

    Args:
        analysis_id: UUID анализа (PK таблицы analysis_results).
        target_col: имя целевого столбца (берётся оркестратором из analyses.target_column).
        task_type: рекомендованный тип задачи (из task_recommendation.task_type_code).
        db: SQLAlchemy-сессия (открывается оркестратором отдельно от HTTP).

    Returns:
        Словарь baseline по контракту baseline-training.md.

    Raises:
        ValueError: если запись Analysis или AnalysisResult не найдена.
    """
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise ValueError(f"Analysis {analysis_id} not found")
    dataset = analysis.dataset

    df = read_dataset_full(Path(dataset.storage_path), dataset.format)

    result = db.get(AnalysisResult, analysis_id)
    if result is None:
        raise ValueError(f"AnalysisResult for analysis {analysis_id} not found")
    meta = result.meta_features or {}

    leakage_cols = _get_leakage_columns(db, analysis_id)

    return train_baseline_from_df(df, meta, leakage_cols, target_col, task_type)
