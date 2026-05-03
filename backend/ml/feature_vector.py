"""
Векторизация meta-features для мета-классификатора и embedding'а в pgvector.

Этот модуль — мостик между профайлером (`app.services.profiler.compute_meta_features`,
выдаёт большой словарь со скалярами, словарями и распределениями) и численными
алгоритмами Слоя 2: мета-классификатором (Random Forest на векторе) и
поиском похожих датасетов (косинусная мера в pgvector). Из всего профиля сюда
попадают только **скалярные** признаки в фиксированном порядке —
`CANONICAL_FEATURE_ORDER`.

Принципы выбора признаков (см. Phase 2 уточнения и `recommender-ml.md`):
- ВКЛЮЧАЕМ: размерности датасета, доли пропусков и дубликатов, агрегаты
  по числовым колонкам (моменты, нормальность, выбросы), агрегаты по target
  (imbalance, энтропия классов, асимметрия), корреляции и Mutual Information.
- ИСКЛЮЧАЕМ: `target_kind` — это эвристика профайлера (`infer_target_kind`),
  использовать её как фичу было бы leakage'ом по отношению к самой задаче
  Слоя 2 (он бы тривиально разрешался, f1_macro≈1.0 искусственно).
- ИСКЛЮЧАЕМ: per-column словари (`*_by_column`), списки колонок
  (`high_cardinality_cols` и т.п.), `distributions`, `correlation_matrix` —
  это вектора переменной размерности и UI-данные, в фиксированный embedding
  не помещаются.
- ВКЛЮЧАЕМ ОДНУ ПРОИЗВОДНУЮ: `features_to_rows_ratio = n_cols / max(n_rows, 1)`.
  Сигнал «high-dimensional vs wide» полезен и не является leakage'ом. Других
  производных не добавляем — дверь в feature engineering закрыта.

Стабильность порядка критична: `CANONICAL_FEATURE_ORDER` фиксирует индексы фичей
в векторе. Если порядок изменится после того, как каталог `external_datasets`
посеян, все existing-embedding'и в БД станут несовместимыми с пользовательскими.
Поэтому в Phase 3 есть тест `test_canonical_feature_order_hash_stable`, который
ловит молчаливое изменение порядка через хеш списка ключей.

См. `.knowledge/methods/recommender-ml.md`, разделы «Сбор обучающей выборки»
и «Применение модели».
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


# Размерность embedding'а в pgvector. Embedding'и собираются как
# CANONICAL_FEATURE_ORDER → StandardScaler.transform → padding нулями до 128.
EMBEDDING_DIM = 128


# Ключ "__features_to_rows_ratio__" — производная (см. модульный docstring).
# Все остальные ключи — прямо из выхода `compute_meta_features`.
DERIVED_FEATURES_TO_ROWS_RATIO = "__features_to_rows_ratio__"


# Итерация выбора признаков (Phase 2):
#
# Шаг 1 (18 фичей, отвергнуто): включали target_imbalance_ratio,
# target_class_entropy, target_skewness. Профайлер выдаёт их `None` для
# регрессионных датасетов и not-None для классификационных, поэтому замена
# None→0.0 в `meta_features_to_vector` превращала их в бинарный индикатор
# `target_kind == "categorical"` — косвенный leakage, эквивалентный включению
# самого `target_kind` (см. Q1 Phase 2). CV f1_macro = 0.987 был искусственным.
#
# Шаг 2 (15 фичей, недостаточно): убрали три leak-фичи. CV f1_macro = 0.796 —
# в пределах Done-порога ± std, но BINARY/MULTICLASS плохо различались (нет
# признака, кодирующего «число классов в target»).
#
# Шаг 3 (16 фичей, текущий): добавили `target_n_unique` — continuous
# дескриптивная статистика (`series.nunique()`), вычисляемая независимо от
# `target_kind`. Это canonical simple meta-feature по Vanschoren 2018 §3.2,
# используемая в meta-learning literature как базовый дескриптор. Не leakage:
# то что Слой 1 правил тоже опирается на «число уникальных значений target»
# не делает этот признак запрещённым для Слоя 2 — оба слоя имеют право
# опираться на одни и те же базовые описания данных, разница в том что
# Слой 2 учится на их комбинациях.
#
# `target_correlation_max` и `target_mutual_information_*` оставлены: они
# вычисляются и для regression (`mutual_info_regression`), и для classification
# (`mutual_info_classif`), не None ни в одном случае.
CANONICAL_FEATURE_ORDER: list[str] = [
    # --- структура датасета ---
    "n_rows",
    "n_cols",
    "memory_mb",
    DERIVED_FEATURES_TO_ROWS_RATIO,
    # --- пропуски и дубликаты ---
    "total_missing_pct",
    "max_col_missing_pct",
    "duplicate_rows_pct",
    # --- агрегаты по числовым колонкам ---
    "mean_skewness",
    "mean_kurtosis",
    "normality_test_pvalue",
    "outliers_pct",
    # --- описание целевой переменной (continuous-описание, не индикатор) ---
    "target_n_unique",
    # --- связи признаков с целью (вычисляются и для regression, и для classification) ---
    "max_abs_correlation",
    "target_correlation_max",
    "target_mutual_information_max",
    "target_mutual_information_mean",
]


def meta_features_to_vector(meta: dict[str, Any]) -> np.ndarray:
    """
    Собирает скалярный вектор из словаря meta-features в каноническом порядке.

    Алгоритм:
    1. Идём по `CANONICAL_FEATURE_ORDER` слева направо.
    2. Для производных ключей (например, `features_to_rows_ratio`) считаем
       значение из других meta-features.
    3. Для прямых ключей берём `meta.get(key)`.
    4. None и нечисловые/нефинитные значения заменяются на 0.0
       (требование sklearn: на StandardScaler.fit нельзя подавать NaN).

    Замена None на 0.0 не идеальна, но в нашей задаче ведёт себя осмысленно:
    если у датасета нет target (clustering), все `target_*`-поля = None → 0.0,
    и embedding клиники чётко отличается от классификации/регрессии. Для
    обучения мета-классификатора clustering-записи всё равно отфильтрованы
    (см. `train_meta_classifier.py`), так что 0.0 у target_* там не появляется.

    Args:
        meta: словарь meta-features из `compute_meta_features`.

    Returns:
        Одномерный массив `np.float64` длины `len(CANONICAL_FEATURE_ORDER)`.
    """
    n_rows = _safe_float(meta.get("n_rows"))
    n_cols = _safe_float(meta.get("n_cols"))

    vector: list[float] = []
    for key in CANONICAL_FEATURE_ORDER:
        if key == DERIVED_FEATURES_TO_ROWS_RATIO:
            value = n_cols / max(n_rows, 1.0)
        else:
            value = _safe_float(meta.get(key))
        vector.append(value)
    return np.asarray(vector, dtype=np.float64)


def compute_feature_order_hash() -> str:
    """
    Детерминированный хеш списка `CANONICAL_FEATURE_ORDER`.

    Используется в тесте стабильности (`test_canonical_feature_order_hash_stable`,
    Phase 3). Если кто-то добавит/удалит/переставит ключ, хеш изменится, и
    тест громко скажет «эмбеддинги в БД несовместимы — пересей каталог».

    Returns:
        Hex SHA-256 от ','.join(CANONICAL_FEATURE_ORDER).
    """
    payload = ",".join(CANONICAL_FEATURE_ORDER).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_float(value: Any) -> float:
    """
    Безопасное приведение к float: None, NaN, ±inf и строки → 0.0.
    Внутренний хелпер, для внешнего использования не предназначен.
    """
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result
