"""
Сервис рекомендации типа ML-задачи: гибрид Слоя 1 (правила) и Слоя 2 (Random Forest).

Главная точка входа — `recommend_task(meta, target_column, active_quality_flags=None)`.
Возвращает `TaskRecommendation` с финальным task_type_code, confidence, источником
решения и текстовым объяснением.

Алгоритм гибрида (см. `.knowledge/methods/recommender-ml.md`, раздел «Логика гибрида»):

    rules_result = apply_rules(meta, target_column, active_quality_flags)
    model, scaler = _load_meta_classifier_safe()

    # Случай 1: модель недоступна — graceful degradation
    if model is None or scaler is None:
        return TaskRecommendation(source="rules", + пометка в explanation)

    # Случай 2: правила уверены и не требуют ML
    if rules_result.confidence >= 0.7 and not rules_result.requires_ml:
        return TaskRecommendation(source="rules", task_type_code из rules_result)

    # Случай 3: вызываем Слой 2
    ml_class, ml_confidence, ml_probas = _apply_ml_classifier(...)
    source = "hybrid" if rules_result.confidence > 0 else "ml"
    return TaskRecommendation(source, task_type_code=ml_class, ...)

Слой 1 (правила) реализует дерево решений из `recommender-rules.md`:
- Ветка 1: target отсутствует → CLUSTERING (или CLUSTERING + DIMENSIONALITY_REDUCTION)
- Ветка 3: numeric target → BINARY/REGRESSION/делегирование по target_n_unique
- Ветка 4: categorical target → BINARY/MULTICLASS/NOT_READY/делегирование
- Ветка 5: критические флаги качества → добавляются к applied_rules как предупреждения

Ветка 2 («TARGET_NOT_USABLE для string/datetime») как самостоятельная не реализуется:
профайлер выдаёт `target_kind ∈ {"categorical", "regression", None}`, и string/datetime
target по эвристике `infer_target_kind` попадают в "categorical". Если кардинальность
высокая — поглощаются Веткой 4 (TARGET_LOOKS_LIKE_ID → NOT_READY), что семантически
эквивалентно «требует предобработки». Расхождение с методичкой зафиксировано для
обновления через update-knowledge в Phase 9.

Слой 2 (ML) — обученная RandomForest из `backend/ml/models/meta_classifier.pkl`.
Загружается лениво и кешируется; при отсутствии файла — graceful degradation
к чистым правилам (см. `recommender-ml.md`, раздел «Fallback при недоступности модели»).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.schemas.task_recommendation import (
    AppliedRule,
    RulesResult,
    TaskRecommendation,
)
from ml.feature_vector import meta_features_to_vector


logger = logging.getLogger(__name__)


# Пути к артефактам Слоя 2. Скрипт обучения (`ml/train_meta_classifier.py`)
# сохраняет их сюда; в production эти файлы коммитятся в образ через `make seed-all`.
ML_MODELS_DIR = Path("ml/models")
META_CLASSIFIER_PATH = ML_MODELS_DIR / "meta_classifier.pkl"
SCALER_PATH = ML_MODELS_DIR / "scaler.pkl"


# Порог уверенности правил, выше которого ML не вызывается (если правила не
# просили делегирования). См. `recommender-ml.md`, раздел «Логика гибрида».
RULES_CONFIDENCE_THRESHOLD = 0.7


# Диапазон n_unique для срабатывания numeric-discrete bridge (см.
# `_should_redirect_to_numeric_branch`). Нижняя граница 3 — bridge не
# конфликтует с n_unique<=2 (BINARY); верхняя 20 — выше уже семантически
# регрессия, и `infer_target_kind` корректно вернёт "regression" сам
# (порог профайлера max(20, 5%·n_rows)).
NUMERIC_BRIDGE_MIN_UNIQUE = 3
NUMERIC_BRIDGE_MAX_UNIQUE = 20


def _is_numeric_label(value: Any) -> bool:
    """
    True если строковый ключ target_value_counts парсится как конечное число.

    Используется numeric-discrete bridge: метки `"0"`, `"1"`, `"2.5"` —
    численные; `"setosa"`, `"male"`, но и `"inf"`/`"nan"` — НЕ численные
    (последние два формально парсятся `float()`, но не дают информации
    о порядке шкалы и не должны переинтерпретироваться как regression).
    """
    try:
        parsed = float(str(value).strip())
    except (ValueError, TypeError):
        return False
    return not (math.isinf(parsed) or math.isnan(parsed))


def _should_redirect_to_numeric_branch(
    *,
    target_kind: str | None,
    target_n_unique: int | None,
    target_value_counts: dict[str, int] | None,
) -> bool:
    """
    Признаки numeric-discrete target, ошибочно классифицированного
    профайлером как categorical.

    Корневая причина: эвристика `infer_target_kind` использует порог
    `max(20, 5%·n_rows)` — для маленьких датасетов (Iris, n=150) это
    значит «categorical при n_unique ≤ 20», что съедает все
    числовые-discrete целевые с малой кардинальностью (target=0/1/2),
    оставляя ветку 3 (numeric) недостижимой и блокируя путь
    AMBIGUOUS_NUMERIC_TARGET → Слой 2 ML.

    Bridge срабатывает только когда target_kind="categorical",
    `n_unique` в диапазоне `[NUMERIC_BRIDGE_MIN_UNIQUE,
    NUMERIC_BRIDGE_MAX_UNIQUE]`, и **все** ключи `target_value_counts`
    парсятся как конечные числа. На string-target ('setosa', 'male')
    bridge не срабатывает — ветка 4 сохраняет своё поведение.

    Архитектурно это «фикс симптома» в recommender'е; правильный
    «фикс источника» — расширение `infer_target_kind` в профайлере
    (Вариант B, отложен после защиты).
    """
    if target_kind != "categorical":
        return False
    if target_n_unique is None or not target_value_counts:
        return False
    if not (
        NUMERIC_BRIDGE_MIN_UNIQUE
        <= target_n_unique
        <= NUMERIC_BRIDGE_MAX_UNIQUE
    ):
        return False
    return all(_is_numeric_label(k) for k in target_value_counts.keys())


# Коды критических флагов из `quality_checker.py`, которые добавляются как
# предупреждения в applied_rules. См. recommender-rules.md, Ветка 5.
CRITICAL_QUALITY_FLAGS = {
    "TARGET_MISSING": "В целевом столбце пропуски — соответствующие наблюдения "
                      "будут отброшены при обучении.",
    "LEAKAGE_SUSPICION": "Подозрение на утечку данных: один из признаков "
                         "слишком сильно связан с target.",
    "SMALL_DATASET": "Мало данных — оценки качества модели могут иметь "
                     "большую дисперсию.",
}


# =============================================================================
#                       1. СЛОЙ 1: ПРАВИЛА
# =============================================================================


def apply_rules(
    meta: dict[str, Any],
    target_column: str | None,
    active_quality_flags: list[str] | None = None,
) -> RulesResult:
    """
    Детерминированное дерево решений Слоя 1.

    Алгоритм веток — точно по `.knowledge/methods/recommender-rules.md`:
    1. target_column is None → CLUSTERING (с DIMENSIONALITY_REDUCTION-меткой при n_features > 10)
    2. target_kind == "regression" → BINARY/REGRESSION/AMBIGUOUS по target_n_unique
    3. target_kind == "categorical" → BINARY/MULTICLASS/NOT_READY/HIGH_CARDINALITY
    4. Критические quality-флаги добавляются как applied_rules без изменения task_type

    Args:
        meta: словарь meta-features из `compute_meta_features()`.
        target_column: имя целевого столбца или None.
        active_quality_flags: коды сработавших правил качества из quality_checker
            (опционально, обычно передаётся из analysis_service в Phase 6).

    Returns:
        `RulesResult` с финальным task_type_code (от правил), confidence,
        флагом `requires_ml` и списком сработавших правил.
    """
    n_rows = int(meta.get("n_rows") or 0)
    n_cols = int(meta.get("n_cols") or 0)
    target_kind: str | None = meta.get("target_kind")
    target_n_unique: int | None = meta.get("target_n_unique")
    target_imbalance_ratio: float | None = meta.get("target_imbalance_ratio")
    target_skewness: float | None = meta.get("target_skewness")
    target_value_counts: dict[str, int] | None = meta.get("target_value_counts")

    # Ветка 1: target не указан.
    if target_column is None or target_kind is None:
        # n_features = n_cols (без target, потому что target отсутствует).
        if n_cols > 10:
            return RulesResult(
                task_type_code="CLUSTERING",
                confidence=0.8,
                requires_ml=False,
                applied_rules=[
                    AppliedRule(
                        code="NO_TARGET_HIGH_DIM",
                        description=(
                            "Целевая переменная не указана и признаков больше 10 — "
                            "рекомендуется снижение размерности (PCA/UMAP) перед "
                            "кластеризацией (KMeans/DBSCAN)."
                        ),
                    ),
                    AppliedRule(
                        code="DIMENSIONALITY_REDUCTION",
                        description=(
                            "Вспомогательный шаг: уменьшить число признаков до 2-5 "
                            "методом главных компонент или нелинейным embedding'ом "
                            "перед запуском алгоритмов кластеризации."
                        ),
                    ),
                ],
            )
        return RulesResult(
            task_type_code="CLUSTERING",
            confidence=0.9,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="NO_TARGET_LOW_DIM",
                    description=(
                        "Целевая переменная не указана, размерность невысокая — "
                        "подойдёт прямая кластеризация (KMeans/DBSCAN)."
                    ),
                )
            ],
        )

    # Numeric-discrete bridge: target с малой кардинальностью и числовыми
    # метками профайлер ошибочно помечает categorical (см. docstring
    # `_should_redirect_to_numeric_branch`). Перенаправляем такие случаи
    # в Ветку 3 — это делает достижимым путь AMBIGUOUS_NUMERIC_TARGET → Слой 2.
    if _should_redirect_to_numeric_branch(
        target_kind=target_kind,
        target_n_unique=target_n_unique,
        target_value_counts=target_value_counts,
    ):
        target_kind = "regression"

    # Ветка 3: numeric target.
    if target_kind == "regression":
        return _apply_numeric_target_rules(
            target_n_unique=target_n_unique,
            target_skewness=target_skewness,
        )

    # Ветка 4: categorical target.
    if target_kind == "categorical":
        return _apply_categorical_target_rules(
            target_n_unique=target_n_unique,
            target_imbalance_ratio=target_imbalance_ratio,
            target_value_counts=target_value_counts,
            n_rows=n_rows,
        )

    # Сюда не должно прилетать (target_kind ∈ {"categorical", "regression", None}).
    # Но если профайлер вернул что-то непредвиденное — деградируем безопасно.
    return RulesResult(
        task_type_code="NOT_READY",
        confidence=0.5,
        requires_ml=False,
        applied_rules=[
            AppliedRule(
                code="UNKNOWN_TARGET_KIND",
                description=f"Неизвестный target_kind: {target_kind!r}. "
                            "Требуется ручная проверка.",
            )
        ],
    )


def _apply_numeric_target_rules(
    *,
    target_n_unique: int | None,
    target_skewness: float | None,
) -> RulesResult:
    """Ветка 3 из recommender-rules.md: target числовой."""
    n_unique = target_n_unique or 0

    if n_unique <= 2:
        return RulesResult(
            task_type_code="BINARY_CLASSIFICATION",
            confidence=0.95,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="NUMERIC_BINARY_TARGET",
                    description=(
                        f"Целевая переменная числовая с {n_unique} уникальными "
                        "значениями — это бинарная классификация (например, 0/1)."
                    ),
                )
            ],
        )

    if n_unique <= 10:
        # Пограничный случай: 3-10 уникальных численных значений могут быть и
        # многоклассовой классификацией, и регрессией. Делегируем в Слой 2.
        return RulesResult(
            task_type_code="MULTICLASS_CLASSIFICATION",
            confidence=0.5,
            requires_ml=True,
            applied_rules=[
                AppliedRule(
                    code="AMBIGUOUS_NUMERIC_TARGET",
                    description=(
                        f"Целевая переменная числовая, всего {n_unique} уникальных "
                        "значений — пограничный случай между многоклассовой "
                        "классификацией и регрессией. Решение делегировано "
                        "ML-классификатору Слоя 2."
                    ),
                )
            ],
        )

    if n_unique <= 20:
        # Если распределение симметрично — скорее MULTICLASS; иначе делегируем.
        if target_skewness is not None and abs(target_skewness) < 1.0:
            return RulesResult(
                task_type_code="MULTICLASS_CLASSIFICATION",
                confidence=0.7,
                requires_ml=False,
                applied_rules=[
                    AppliedRule(
                        code="NUMERIC_TARGET_SYMMETRIC_FEW_VALUES",
                        description=(
                            f"Целевая переменная числовая, {n_unique} значений, "
                            f"распределение симметричное (|skewness|={abs(target_skewness):.2f} < 1) — "
                            "вероятно, многоклассовая классификация."
                        ),
                    )
                ],
            )
        return RulesResult(
            task_type_code="REGRESSION",
            confidence=0.5,
            requires_ml=True,
            applied_rules=[
                AppliedRule(
                    code="AMBIGUOUS_NUMERIC_TARGET",
                    description=(
                        f"Целевая переменная числовая, {n_unique} значений, "
                        "распределение скошенное — пограничный случай. "
                        "Решение делегировано ML-классификатору Слоя 2."
                    ),
                )
            ],
        )

    # n_unique > 20 — почти наверняка регрессия.
    return RulesResult(
        task_type_code="REGRESSION",
        confidence=0.95,
        requires_ml=False,
        applied_rules=[
            AppliedRule(
                code="NUMERIC_TARGET_MANY_VALUES",
                description=(
                    f"Целевая переменная числовая с {n_unique} уникальными "
                    "значениями — задача регрессии."
                ),
            )
        ],
    )


def _apply_categorical_target_rules(
    *,
    target_n_unique: int | None,
    target_imbalance_ratio: float | None,
    target_value_counts: dict[str, int] | None,
    n_rows: int,
) -> RulesResult:
    """Ветка 4 из recommender-rules.md: target категориальный."""
    n_classes = target_n_unique or 0

    if n_classes < 2:
        return RulesResult(
            task_type_code="NOT_READY",
            confidence=0.99,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="ONLY_ONE_CLASS",
                    description=(
                        "В целевом столбце только один класс — нельзя обучить "
                        "классификатор. Проверьте, что target указан правильно "
                        "и в данных есть несколько классов."
                    ),
                )
            ],
        )

    if n_classes == 2:
        ratio = target_imbalance_ratio or 0.0
        if ratio > 10.0:
            return RulesResult(
                task_type_code="BINARY_CLASSIFICATION",
                confidence=0.9,
                requires_ml=False,
                applied_rules=[
                    AppliedRule(
                        code="BINARY_IMBALANCED",
                        description=(
                            f"Бинарная классификация с дисбалансом {ratio:.1f}:1 "
                            "(порог 10:1). Используйте F1/ROC-AUC и техники "
                            "балансировки (oversampling, class_weight)."
                        ),
                    )
                ],
            )
        return RulesResult(
            task_type_code="BINARY_CLASSIFICATION",
            confidence=0.95,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="BINARY_BALANCED",
                    description=(
                        f"Целевая переменная категориальная, 2 класса, дисбаланс "
                        f"{ratio:.1f}:1 (≤ 10:1) — стандартная бинарная классификация."
                    ),
                )
            ],
        )

    if 3 <= n_classes <= 20:
        min_class_size = (
            min(target_value_counts.values()) if target_value_counts else None
        )
        if min_class_size is not None and min_class_size < 50:
            return RulesResult(
                task_type_code="MULTICLASS_CLASSIFICATION",
                confidence=0.85,
                requires_ml=False,
                applied_rules=[
                    AppliedRule(
                        code="MULTICLASS",
                        description=(
                            f"Многоклассовая классификация на {n_classes} классов. "
                            f"Минимальный класс — всего {min_class_size} объектов "
                            "(порог 50): используйте f1_macro и stratified-split."
                        ),
                    )
                ],
            )
        return RulesResult(
            task_type_code="MULTICLASS_CLASSIFICATION",
            confidence=0.95,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="MULTICLASS",
                    description=(
                        f"Многоклассовая классификация на {n_classes} классов "
                        "с достаточным размером каждого класса."
                    ),
                )
            ],
        )

    # n_classes > 20: возможно target похож на ID.
    target_cardinality_ratio = n_classes / max(n_rows, 1)
    if target_cardinality_ratio > 0.5:
        return RulesResult(
            task_type_code="NOT_READY",
            confidence=0.8,
            requires_ml=False,
            applied_rules=[
                AppliedRule(
                    code="TARGET_LOOKS_LIKE_ID",
                    description=(
                        f"Целевая переменная имеет {n_classes} уникальных значений "
                        f"при {n_rows} строках (cardinality ratio "
                        f"{target_cardinality_ratio:.1%} > 50%) — похоже, это "
                        "идентификатор, а не метка класса."
                    ),
                )
            ],
        )
    # Высокая кардинальность, но не похоже на ID — делегируем в Слой 2.
    return RulesResult(
        task_type_code="MULTICLASS_CLASSIFICATION",
        confidence=0.6,
        requires_ml=True,
        applied_rules=[
            AppliedRule(
                code="HIGH_CARDINALITY_MULTICLASS",
                description=(
                    f"Многоклассовая классификация на {n_classes} классов — "
                    "много для надёжного обучения. Решение требует ML-уточнения."
                ),
            )
        ],
    )


def _attach_quality_warnings(
    rules_result: RulesResult,
    active_quality_flags: list[str] | None,
) -> None:
    """
    Дополняет `applied_rules` предупреждениями из критических quality-флагов.
    Меняет `rules_result` in-place; task_type_code и confidence не трогает.
    """
    if not active_quality_flags:
        return
    for code in active_quality_flags:
        if code in CRITICAL_QUALITY_FLAGS:
            rules_result.applied_rules.append(
                AppliedRule(
                    code=code,
                    description=CRITICAL_QUALITY_FLAGS[code],
                )
            )


# =============================================================================
#                       2. СЛОЙ 2: ML-КЛАССИФИКАТОР
# =============================================================================


# Module-level кеш модели и scaler'а. Загружаются лениво при первом вызове
# `_load_meta_classifier_safe`, далее переиспользуются. Кеш сбрасывается
# через `force_reload=True` (используется в тестах для monkeypatch'а путей).
_MODEL: Any | None = None
_SCALER: Any | None = None
_LOADED: bool = False


def _load_meta_classifier_safe(
    *,
    force_reload: bool = False,
) -> tuple[Any | None, Any | None]:
    """
    Безопасная загрузка обученной модели и scaler'а из `backend/ml/models/`.

    При отсутствии любого из файлов возвращает `(None, None)` и пишет warning
    в лог — это нормальный режим работы до первого `make train-meta`. Слой 1
    в этом случае работает как обычно (graceful degradation).

    Args:
        force_reload: если True — игнорировать кеш и перечитать файлы заново.
            Нужно тестам, чтобы monkeypatch'нуть пути к моделям.

    Returns:
        (model, scaler) либо (None, None).
    """
    global _MODEL, _SCALER, _LOADED
    if _LOADED and not force_reload:
        return _MODEL, _SCALER

    try:
        if not META_CLASSIFIER_PATH.exists() or not SCALER_PATH.exists():
            logger.warning(
                "Meta-classifier or scaler not found in %s — falling back to "
                "rules-only mode. Run `make train-meta` to train the model.",
                ML_MODELS_DIR,
            )
            _MODEL, _SCALER = None, None
        else:
            _MODEL = joblib.load(META_CLASSIFIER_PATH)
            _SCALER = joblib.load(SCALER_PATH)
            logger.info("Meta-classifier loaded from %s", META_CLASSIFIER_PATH)
    except Exception:  # noqa: BLE001 — любой сбой загрузки = graceful degradation
        logger.exception("Failed to load meta-classifier; falling back to rules-only")
        _MODEL, _SCALER = None, None

    _LOADED = True
    return _MODEL, _SCALER


def _apply_ml_classifier(
    meta: dict[str, Any],
    model: Any,
    scaler: Any,
) -> tuple[str, float, dict[str, float]]:
    """
    Применяет обученную RandomForest-модель к meta-features.

    Алгоритм:
    1. Векторизуем meta через `meta_features_to_vector` (16 признаков в каноническом порядке).
    2. Применяем StandardScaler (тот же, что использовался при обучении).
    3. Получаем `predict_proba`, выбираем класс с максимальной вероятностью.

    Returns:
        Кортеж (best_class, best_proba, all_probas), где all_probas — словарь
        {class_name: probability} для всех 3 классов модели.
    """
    vector = meta_features_to_vector(meta).reshape(1, -1)
    scaled = scaler.transform(vector)
    probas = model.predict_proba(scaled)[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(probas))
    return (
        str(classes[best_idx]),
        float(probas[best_idx]),
        {str(cls): float(p) for cls, p in zip(classes, probas)},
    )


# =============================================================================
#                       3. ОБЪЯСНЕНИЕ
# =============================================================================


_TASK_TYPE_NAMES_RU = {
    "BINARY_CLASSIFICATION": "бинарная классификация",
    "MULTICLASS_CLASSIFICATION": "многоклассовая классификация",
    "REGRESSION": "регрессия",
    "CLUSTERING": "кластеризация",
    "NOT_READY": "данные не готовы для ML",
}


def _build_explanation(
    rules_result: RulesResult,
    *,
    final_task_type: str,
    source: str,
    ml_class: str | None = None,
    ml_confidence: float | None = None,
    model_unavailable: bool = False,
) -> str:
    """
    Собирает русский текст «Почему такая рекомендация» для UI.

    Структура:
    - Главный вывод: «Рекомендуемый тип задачи — <русское имя>».
    - Сработавшие правила Слоя 1 (если есть): по строке на правило.
    - Если модель применялась — мягкая формулировка про confidence.
    - Если модель недоступна — пометка «используются только правила».
    """
    parts: list[str] = []

    final_name_ru = _TASK_TYPE_NAMES_RU.get(final_task_type, final_task_type)
    parts.append(f"Рекомендуемый тип задачи: {final_name_ru}.")

    if rules_result.applied_rules:
        parts.append("Сработавшие правила:")
        for rule in rules_result.applied_rules:
            parts.append(f"• {rule.description}")

    if source in {"ml", "hybrid"} and ml_class is not None:
        ml_name_ru = _TASK_TYPE_NAMES_RU.get(ml_class, ml_class)
        pct = (ml_confidence or 0.0) * 100
        parts.append(
            f"Модель Слоя 2 склоняется к типу «{ml_name_ru}» с вероятностью "
            f"{pct:.0f}%. Окончательное решение остаётся за вами."
        )

    if model_unavailable:
        parts.append(
            "На текущем этапе используются только правила; ML-уточнение будет "
            "доступно после обучения модели (см. `make train-meta`)."
        )

    return "\n".join(parts)


# =============================================================================
#                       4. ГЛАВНАЯ ТОЧКА ВХОДА
# =============================================================================


def recommend_task(
    meta: dict[str, Any],
    target_column: str | None,
    active_quality_flags: list[str] | None = None,
) -> TaskRecommendation:
    """
    Главная функция: гибрид Слоя 1 (правила) + Слоя 2 (ML).

    Args:
        meta: словарь meta-features из `compute_meta_features()`.
        target_column: имя целевого столбца или None.
        active_quality_flags: коды сработавших правил качества (опционально).
            Используется для дополнения applied_rules критическими флагами
            (TARGET_MISSING/LEAKAGE_SUSPICION/SMALL_DATASET) согласно Ветке 5
            recommender-rules.md.

    Returns:
        `TaskRecommendation`, готовая к сериализации в JSONB через `model_dump()`.
    """
    rules_result = apply_rules(meta, target_column, active_quality_flags)
    _attach_quality_warnings(rules_result, active_quality_flags)

    model, scaler = _load_meta_classifier_safe()

    # Случай 1: модель недоступна — graceful degradation на чистые правила.
    if model is None or scaler is None:
        return TaskRecommendation(
            task_type_code=rules_result.task_type_code,  # type: ignore[arg-type]
            confidence=rules_result.confidence,
            source="rules",
            applied_rules=rules_result.applied_rules,
            ml_probabilities=None,
            explanation=_build_explanation(
                rules_result,
                final_task_type=rules_result.task_type_code,
                source="rules",
                model_unavailable=True,
            ),
        )

    # Случай 2: правила уверены и не требуют ML.
    if (
        rules_result.confidence >= RULES_CONFIDENCE_THRESHOLD
        and not rules_result.requires_ml
    ):
        return TaskRecommendation(
            task_type_code=rules_result.task_type_code,  # type: ignore[arg-type]
            confidence=rules_result.confidence,
            source="rules",
            applied_rules=rules_result.applied_rules,
            ml_probabilities=None,
            explanation=_build_explanation(
                rules_result,
                final_task_type=rules_result.task_type_code,
                source="rules",
            ),
        )

    # Случай 3: вызываем Слой 2 (либо confidence низкий, либо requires_ml).
    ml_class, ml_confidence, ml_probas = _apply_ml_classifier(meta, model, scaler)
    source = "hybrid" if rules_result.confidence > 0 else "ml"
    return TaskRecommendation(
        task_type_code=ml_class,  # type: ignore[arg-type]
        confidence=ml_confidence,
        source=source,  # type: ignore[arg-type]
        applied_rules=rules_result.applied_rules,
        ml_probabilities=ml_probas,
        explanation=_build_explanation(
            rules_result,
            final_task_type=ml_class,
            source=source,
            ml_class=ml_class,
            ml_confidence=ml_confidence,
        ),
    )
