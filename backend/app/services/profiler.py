"""
Сервис профилирования табличных датасетов: вычисление meta-features.

Модуль реализует первый научный слой системы — расчёт ~30 признакового описания
датасета (meta-features), которые используются:
1. Quality checker'ом для применения 12 правил качества данных.
2. Подбором похожих датасетов в Спринте 3 (через нормализацию meta-features
   в эмбеддинг и косинусную меру в pgvector).
3. Мета-классификатором для рекомендации типа ML-задачи (Спринт 3).

Полное теоретическое обоснование методов и список всех meta-features —
в .knowledge/methods/profiling.md (этот файл войдёт в Главу 2 РПЗ).

Используемые методы и источники:
- Tukey J.W. "Exploratory Data Analysis", 1977 — IQR-метод обнаружения выбросов.
- Shapiro S.S., Wilk M.B. "An analysis of variance test for normality", 1965 —
  тест нормальности.
- Cover T., Thomas J. "Elements of Information Theory", 2006 — Mutual Information.
- Shannon C.E. "A Mathematical Theory of Communication", 1948 — энтропия.

Принципы реализации (см. CLAUDE.md):
- Воспроизводимость: все рандомизированные операции (сэмплирование, MI)
  фиксируются `random_state=42`.
- Объяснимость: каждая meta-feature имеет ясное определение и источник —
  чтобы её можно было защитить на ГЭК.
- Производительность: для df > 50 000 строк применяется сэмплирование, иначе
  тест Шапиро-Уилка и Mutual Information работают слишком долго.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Глобальные параметры профилирования.
SAMPLING_THRESHOLD = 50_000  # после этого числа строк применяем сэмплирование
SAMPLE_SIZE = 50_000
SHAPIRO_MAX_N = 5_000  # ограничение scipy.stats.shapiro
RANDOM_STATE = 42  # для воспроизводимости (требование ТЗ)

# Числовые dtype'ы pandas, для которых считаются статистические meta-features.
NUMERIC_KINDS = {"i", "u", "f"}  # signed/unsigned int, float

# =============================================================================
#                           1. ОБНАРУЖЕНИЕ ВЫБРОСОВ
# =============================================================================


def detect_outliers_iqr(values: np.ndarray) -> np.ndarray:
    """
    Обнаружение выбросов методом межквартильного размаха (IQR) Тьюки.

    Метод: значение считается выбросом, если выходит за пределы
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR], где IQR = Q3 - Q1.
    Робастен к выбросам — использует квартили, а не среднее. Применяется к
    распределениям, не близким к нормальному.

    Источник: Tukey J.W. "Exploratory Data Analysis", 1977.
    См. .knowledge/methods/profiling.md, раздел 1.1.

    Args:
        values: одномерный массив числовых значений (без NaN).

    Returns:
        Булев массив той же длины: True — значение является выбросом.
    """
    if values.size == 0:
        return np.array([], dtype=bool)
    # Q1, Q3 — 25-й и 75-й перцентили; IQR — расстояние между ними.
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values < lower) | (values > upper)


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Обнаружение выбросов методом стандартизированных оценок (Z-score).

    Метод: значение считается выбросом, если |z| > threshold, где z = (x-μ)/σ.
    Применим только к близким к нормальному распределениям; для тяжёлых
    хвостов даёт ложные срабатывания, поэтому в нашей системе z-score
    применяется только после проверки нормальности (Шапиро-Уилк).

    Источник: классический метод математической статистики.
    См. .knowledge/methods/profiling.md, раздел 1.2.

    Args:
        values: одномерный массив числовых значений (без NaN).
        threshold: порог для модуля z-score, по умолчанию 3.

    Returns:
        Булев массив той же длины: True — значение является выбросом.
    """
    if values.size == 0:
        return np.array([], dtype=bool)
    mean = values.mean()
    std = values.std(ddof=0)
    # При нулевом стандартном отклонении все значения одинаковы — выбросов нет.
    if std == 0:
        return np.zeros_like(values, dtype=bool)
    z = (values - mean) / std
    return np.abs(z) > threshold


def count_outliers(values: np.ndarray, is_normal: bool) -> int:
    """
    Подсчёт выбросов с автоматическим выбором метода по результату теста нормальности.

    Логика выбора согласно .knowledge/methods/profiling.md, раздел 1.3:
    - распределение нормальное → Z-score (порог |z|>3);
    - иначе → IQR-метод Тьюки (более робастен).

    Args:
        values: одномерный массив числовых значений (без NaN).
        is_normal: результат теста нормальности (True — распределение нормальное).

    Returns:
        Количество выбросов в массиве.
    """
    if values.size == 0:
        return 0
    mask = detect_outliers_zscore(values) if is_normal else detect_outliers_iqr(values)
    return int(mask.sum())


# =============================================================================
#                       2. ТЕСТ НОРМАЛЬНОСТИ И МОМЕНТЫ
# =============================================================================


def is_normal_shapiro(
    values: np.ndarray,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[bool, float | None]:
    """
    Проверка нормальности распределения тестом Шапиро-Уилка.

    Метод: H0 — выборка извлечена из нормального распределения. Если
    p-value < alpha, гипотеза H0 отвергается → распределение не нормальное.

    Ограничение scipy.stats.shapiro: 3 ≤ n ≤ 5000. Для бóльших выборок
    документация рекомендует Anderson-Darling test или сэмпл; мы выбираем
    случайный сэмпл размером SHAPIRO_MAX_N (фиксированный random_state ради
    воспроизводимости — требование ТЗ). Этот компромисс защитим на ГЭК так:
    в наших данных «зацепить» нетипичные хвосты при сэмпле в 5000 точек
    статистически почти невозможно, при этом ускорение колоссальное.

    Источник: Shapiro S.S., Wilk M.B. "An analysis of variance test for
    normality (complete samples)", Biometrika, 1965.
    См. .knowledge/methods/profiling.md, раздел 2.1.

    Args:
        values: одномерный массив числовых значений (без NaN).
        alpha: уровень значимости, по умолчанию 0.05.
        random_state: seed для воспроизводимого сэмпла при n > SHAPIRO_MAX_N.

    Returns:
        Кортеж (is_normal, p_value). При n < 3 возвращается (False, None) —
        тест не применим, выбросы будем считать через IQR.
    """
    if values.size < 3:
        return False, None
    # На константе тест неприменим (range zero) — scipy печатает warning,
    # а математический смысл нормальности отсутствует. Сразу не нормально.
    if values.std(ddof=0) == 0:
        return False, None
    # Для больших выборок берём случайный сэмпл фиксированного размера —
    # сам тест Шапиро-Уилка определён только до n=5000.
    if values.size > SHAPIRO_MAX_N:
        rng = np.random.default_rng(random_state)
        values = rng.choice(values, size=SHAPIRO_MAX_N, replace=False)
    statistic, p_value = stats.shapiro(values)
    return bool(p_value > alpha), float(p_value)


def compute_skewness(values: np.ndarray) -> float | None:
    """
    Асимметрия распределения (третий стандартизированный момент).

    γ₁ = E[(X-μ)³]/σ³. Близко к 0 — симметричное; > 0 — длинный правый хвост;
    < 0 — длинный левый хвост.

    См. .knowledge/methods/profiling.md, раздел 3.1.

    Args:
        values: одномерный массив числовых значений (без NaN).

    Returns:
        Значение асимметрии или None, если выборка слишком мала / константна.
    """
    if values.size < 3:
        return None
    result = float(stats.skew(values))
    return result if np.isfinite(result) else None


def compute_kurtosis(values: np.ndarray) -> float | None:
    """
    Эксцесс распределения (избыточный, по Фишеру).

    γ₂ = E[(X-μ)⁴]/σ⁴ - 3. Близко к 0 — нормальное распределение;
    > 0 — острая вершина и тяжёлые хвосты; < 0 — плоская вершина.

    См. .knowledge/methods/profiling.md, раздел 3.2.

    Args:
        values: одномерный массив числовых значений (без NaN).

    Returns:
        Значение эксцесса или None для слишком малой / константной выборки.
    """
    if values.size < 4:
        return None
    result = float(stats.kurtosis(values, fisher=True))
    return result if np.isfinite(result) else None


def compute_entropy(values: pd.Series) -> float | None:
    """
    Энтропия Шеннона дискретной случайной величины.

    H(X) = -Σ pᵢ log₂(pᵢ), где pᵢ — относительная частота i-го значения.
    Используется для категориальных признаков:
    - H = 0 → все значения одинаковы (флаг LOW_VARIANCE).
    - H = log₂(k) → все k значений равновероятны (максимальное разнообразие).

    Источник: Shannon C.E. "A Mathematical Theory of Communication", 1948.
    См. .knowledge/methods/profiling.md, раздел 4.1.

    Args:
        values: серия категориальных значений (могут быть NaN — пропускаются).

    Returns:
        Значение энтропии в битах или None, если непустых значений нет.
    """
    clean = values.dropna()
    if clean.empty:
        return None
    counts = clean.value_counts().to_numpy()
    # log2 даёт энтропию в битах — стандартное соглашение для дискретных данных.
    return float(stats.entropy(counts, base=2))


def normalized_entropy(values: pd.Series) -> float | None:
    """
    Нормированная энтропия: H(X) / log₂(unique_count) ∈ [0, 1].

    1.0 — все значения равновероятны; 0.0 — одно значение доминирует.
    Удобна для сравнения признаков с разной кардинальностью.
    См. .knowledge/methods/profiling.md, раздел 4.1.
    """
    h = compute_entropy(values)
    if h is None:
        return None
    n_unique = values.dropna().nunique()
    if n_unique <= 1:
        return 0.0
    return float(h / np.log2(n_unique))


# =============================================================================
#                       3. КОЛОНКИ И ИХ ТИПЫ
# =============================================================================


def categorize_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Группировка колонок по типам для дальнейшего профилирования.

    Возвращает словарь с тремя списками:
    - numeric: int/float колонки (на них считаем выбросы, моменты, корреляции);
    - categorical: object/category/bool (на них считаем энтропию, кардинальность);
    - datetime: datetime64-колонки (отдельная группа, в meta-features используется
      ограниченно — главным образом как «уже распарсенные даты»).

    Args:
        df: входной DataFrame.

    Returns:
        Словарь {numeric: [...], categorical: [...], datetime: [...]}.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind in NUMERIC_KINDS:
            numeric.append(col)
        elif kind in {"M", "m"}:  # datetime64 / timedelta64
            datetime.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime}


def infer_target_kind(series: pd.Series) -> str:
    """
    Эвристическое определение типа целевой переменной.

    Используется для выбора алгоритма Mutual Information и для выбора метрик
    рекомендателем (Спринт 3). Логика:
    - dtype нечисловой → categorical;
    - число уникальных значений ≤ max(20, 5% от длины) → categorical;
    - иначе → regression.

    Args:
        series: серия значений целевой переменной.

    Returns:
        "categorical" или "regression".
    """
    if series.dtype.kind not in NUMERIC_KINDS:
        return "categorical"
    n_unique = series.nunique(dropna=True)
    threshold = max(20, int(0.05 * len(series)))
    return "categorical" if n_unique <= threshold else "regression"


# =============================================================================
#                       4. СЭМПЛИРОВАНИЕ БОЛЬШИХ ДАТАСЕТОВ
# =============================================================================


def maybe_sample(
    df: pd.DataFrame,
    target_col: str | None = None,
    threshold: int = SAMPLING_THRESHOLD,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Сэмплирование датасета для ускорения профилирования.

    Если число строк > threshold, выполняется выборка размером sample_size:
    - стратифицированная (по target_col) — если target указан и категориальный;
    - случайная — иначе.

    Факт сэмплирования логируется в meta-features как
    {"sampled": true, "sample_size": ..., "original_size": ...}, чтобы
    пользователь видел, на каких данных производился расчёт. Это требование
    объяснимости (см. CLAUDE.md, принципы проекта).

    Args:
        df: исходный DataFrame.
        target_col: имя целевого столбца (для стратификации).
        threshold: порог числа строк, после которого включается сэмплирование.
        sample_size: целевой размер сэмпла.
        random_state: seed для воспроизводимости.

    Returns:
        Кортеж (sampled_df, info), где info — словарь с описанием
        проведённого сэмплирования.
    """
    info: dict[str, Any] = {
        "sampled": False,
        "sample_size": len(df),
        "original_size": len(df),
    }
    if len(df) <= threshold:
        return df, info

    if target_col and target_col in df.columns:
        target_kind = infer_target_kind(df[target_col])
        if target_kind == "categorical":
            # Стратифицированный сэмпл через DataFrameGroupBy.sample(frac=...):
            # pandas сэмплирует внутри каждой группы пропорционально её размеру.
            frac = sample_size / len(df)
            sampled = (
                df.groupby(target_col, group_keys=False, observed=True)
                .sample(frac=frac, random_state=random_state)
                .reset_index(drop=True)
            )
        else:
            sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    info.update(sampled=True, sample_size=len(sampled))
    return sampled, info


# =============================================================================
#                       5. МЕТА-ФИЧИ ПО ГРУППАМ
# =============================================================================


def compute_structural_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Структурные meta-features: размеры, типы колонок, объём в памяти.

    Возвращает поля n_rows, n_cols, dtype_counts (распределение dtypes),
    memory_mb (объём в памяти в мегабайтах).
    """
    dtype_counts: dict[str, int] = {}
    for dtype in df.dtypes:
        key = str(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "dtype_counts": dtype_counts,
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
    }


def compute_missing_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Meta-features по пропускам.

    - total_missing_pct: общая доля пропущенных ячеек.
    - max_col_missing_pct: максимальная доля пропусков среди колонок.
    - missing_by_column: словарь {колонка: доля пропусков} — нужен для
      bar chart в UI и срабатывания правила HIGH_MISSING.
    """
    total_cells = df.size or 1
    missing_by_column = (df.isna().sum() / max(len(df), 1)).to_dict()
    missing_by_column = {k: float(v) for k, v in missing_by_column.items()}
    return {
        "total_missing_pct": float(df.isna().sum().sum() / total_cells),
        "max_col_missing_pct": float(max(missing_by_column.values(), default=0.0)),
        "missing_by_column": missing_by_column,
    }


def compute_duplicate_features(df: pd.DataFrame) -> dict[str, Any]:
    """Доля полных дубликатов строк (для срабатывания правила DUPLICATES)."""
    if len(df) == 0:
        return {"duplicate_rows_pct": 0.0}
    return {"duplicate_rows_pct": float(df.duplicated().sum() / len(df))}


def compute_numeric_features(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """
    Meta-features по числовым колонкам.

    Для каждой числовой колонки считаем: skewness, kurtosis, факт нормальности
    (Шапиро-Уилк), долю выбросов (Z-score / IQR в зависимости от нормальности).

    Агрегаты по всем числовым колонкам:
    - mean_skewness, mean_kurtosis — средние моменты (используются как
      признаки для мета-классификатора в Спринте 3).
    - normality_test_pvalue — медиана p-value, агрегат «насколько типично
      данные похожи на нормальные».
    - outliers_pct — суммарная доля выбросов по всем числовым колонкам.

    Дополнительно: per-column outliers и low_variance_cols — для UI и
    срабатывания правил OUTLIERS / LOW_VARIANCE.
    """
    skewness_values: list[float] = []
    kurtosis_values: list[float] = []
    pvalues: list[float] = []
    outliers_by_column: dict[str, float] = {}
    low_variance_cols: list[str] = []
    total_outliers = 0
    total_values = 0

    for col in numeric_cols:
        clean = df[col].dropna().to_numpy(dtype=float, na_value=np.nan)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            outliers_by_column[col] = 0.0
            continue

        skew = compute_skewness(clean)
        kurt = compute_kurtosis(clean)
        if skew is not None:
            skewness_values.append(skew)
        if kurt is not None:
            kurtosis_values.append(kurt)

        is_normal, p_value = is_normal_shapiro(clean)
        if p_value is not None:
            pvalues.append(p_value)

        outliers = count_outliers(clean, is_normal)
        outliers_by_column[col] = float(outliers / clean.size)
        total_outliers += outliers
        total_values += clean.size

        # Низкая вариативность по коэффициенту вариации: std/|mean| < 0.01.
        # Если mean = 0 и std = 0 — колонка константная, тоже считаем low_variance.
        mean = float(np.abs(clean.mean()))
        std = float(clean.std(ddof=0))
        if std == 0:
            low_variance_cols.append(col)
        elif mean > 0 and std / mean < 0.01:
            low_variance_cols.append(col)

    return {
        "mean_skewness": float(np.mean(skewness_values)) if skewness_values else None,
        "mean_kurtosis": float(np.mean(kurtosis_values)) if kurtosis_values else None,
        "normality_test_pvalue": float(np.median(pvalues)) if pvalues else None,
        "outliers_pct": float(total_outliers / total_values) if total_values else 0.0,
        "outliers_by_column": outliers_by_column,
        "low_variance_numeric_cols": low_variance_cols,
    }


def compute_categorical_features(
    df: pd.DataFrame, categorical_cols: list[str]
) -> dict[str, Any]:
    """
    Meta-features по категориальным колонкам.

    Для каждой считаем cardinality_ratio = unique/total и нормированную
    энтропию. Агрегаты:
    - high_cardinality_cols: список колонок с cardinality > 0.5.
    - low_variance_categorical_cols: колонки с нормированной энтропией < 0.1.
    - cardinality_by_column / entropy_by_column — для UI и срабатывания
      правил HIGH_CARDINALITY / LOW_VARIANCE.
    """
    cardinality_by_column: dict[str, float] = {}
    entropy_by_column: dict[str, float | None] = {}
    high_cardinality: list[str] = []
    low_variance: list[str] = []

    n_rows = max(len(df), 1)
    for col in categorical_cols:
        n_unique = int(df[col].nunique(dropna=True))
        ratio = n_unique / n_rows
        cardinality_by_column[col] = float(ratio)
        entropy = normalized_entropy(df[col])
        entropy_by_column[col] = entropy
        if ratio > 0.5:
            high_cardinality.append(col)
        if entropy is not None and entropy < 0.1:
            low_variance.append(col)

    return {
        "high_cardinality_cols": high_cardinality,
        "low_variance_categorical_cols": low_variance,
        "cardinality_by_column": cardinality_by_column,
        "entropy_by_column": entropy_by_column,
    }


# =============================================================================
#                       6. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ И MUTUAL INFORMATION
# =============================================================================


def compute_target_features(
    df: pd.DataFrame, target_col: str | None
) -> dict[str, Any]:
    """
    Meta-features по целевой переменной.

    Если target не указан — все target_*-поля = None (явно null, не 0 и не -1):
    задача может быть кластеризацией, и MI/imbalance в этом случае
    не определены.

    Для категориального target считаем imbalance_ratio (max/min размер
    класса) и энтропию классов; для регрессионного — асимметрию распределения.

    Args:
        df: DataFrame.
        target_col: имя целевого столбца (или None).

    Returns:
        Словарь с ключами target_kind, target_imbalance_ratio,
        target_class_entropy, target_skewness, target_value_counts.
    """
    result: dict[str, Any] = {
        "target_kind": None,
        "target_imbalance_ratio": None,
        "target_class_entropy": None,
        "target_skewness": None,
        "target_value_counts": None,
    }
    if target_col is None or target_col not in df.columns:
        return result

    series = df[target_col].dropna()
    if series.empty:
        return result

    kind = infer_target_kind(series)
    result["target_kind"] = kind

    if kind == "categorical":
        counts = series.value_counts()
        result["target_value_counts"] = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            result["target_imbalance_ratio"] = float(counts.max() / counts.min())
        result["target_class_entropy"] = float(
            stats.entropy(counts.to_numpy(), base=2)
        )
    else:
        values = series.to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        result["target_skewness"] = compute_skewness(values)

    return result


def compute_mi_with_target(
    df: pd.DataFrame, target_col: str | None
) -> dict[str, Any] | None:
    """
    Mutual Information между каждым признаком и целевой переменной.

    КРИТИЧНО: MI считается ТОЛЬКО парами признак↔target, НЕ между парами
    признаков. Расчёт MI между всеми парами столбцов был бы O(N²) и
    не нужен для наших задач (рекомендация типа задачи + leakage).

    Если target_col отсутствует — функция возвращает None (явно null), а в
    финальных meta-features появится target_mutual_information_max=None.
    Кластеризация — задача без target, MI там не определён.

    Реализация:
    - Категориальные признаки кодируются LabelEncoder (sklearn требует числа).
    - Пропуски в признаках заполняются 0; в target — наблюдения отбрасываются.
    - Алгоритм sklearn выбирается по типу target: mutual_info_classif для
      категориального, mutual_info_regression для числового.

    Источник: Cover T., Thomas J. "Elements of Information Theory", 2006.
    См. .knowledge/methods/profiling.md, раздел 5.3.

    Args:
        df: DataFrame, включая колонку target.
        target_col: имя целевой переменной (или None).

    Returns:
        Словарь {max, mean, by_column} либо None.
    """
    if target_col is None or target_col not in df.columns:
        return None

    target_kind = infer_target_kind(df[target_col])

    # Отбрасываем строки с NaN в target — для них MI не определён.
    mask = df[target_col].notna()
    if mask.sum() < 2:
        return None
    df_local = df[mask]

    feature_cols = [c for c in df_local.columns if c != target_col]
    if not feature_cols:
        return None

    # Подготовка матрицы признаков X: категориальные кодируем LabelEncoder,
    # числовые приводим к float, пропуски заполняем нулями (требование sklearn).
    X = pd.DataFrame(index=df_local.index)
    for col in feature_cols:
        series = df_local[col]
        if series.dtype.kind in NUMERIC_KINDS:
            X[col] = series.fillna(0.0).astype(float)
        else:
            # Перед LabelEncoder заменяем NaN на сентинельную строку,
            # иначе sklearn упадёт на смешанных типах.
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(series.fillna("MISSING").astype(str))

    y = df_local[target_col]
    if target_kind == "categorical":
        y_encoded = LabelEncoder().fit_transform(y.astype(str))
        scores = mutual_info_classif(X, y_encoded, random_state=RANDOM_STATE)
    else:
        scores = mutual_info_regression(
            X, y.astype(float), random_state=RANDOM_STATE
        )

    by_column = {col: float(score) for col, score in zip(feature_cols, scores)}
    return {
        "max": float(np.max(scores)) if len(scores) else 0.0,
        "mean": float(np.mean(scores)) if len(scores) else 0.0,
        "by_column": by_column,
    }


# =============================================================================
#                       7. КОРРЕЛЯЦИИ
# =============================================================================


def compute_correlations(
    df: pd.DataFrame, numeric_cols: list[str], target_col: str | None
) -> dict[str, Any]:
    """
    Матрица корреляций Пирсона между числовыми признаками + связь с target.

    Возвращает:
    - matrix: словарь {col_i: {col_j: r}} для отрисовки heatmap (только числовые).
    - max_abs_correlation: максимум |r| среди всех пар (i ≠ j).
    - target_correlation_max: максимум |r| между признаком и числовым target
      (если target числовой). Используется правилом LEAKAGE_SUSPICION.

    Источник: классическая статистика. См. .knowledge/methods/profiling.md,
    раздел 5.1.
    """
    result: dict[str, Any] = {
        "matrix": {},
        "max_abs_correlation": None,
        "target_correlation_max": None,
        "target_correlation_by_column": None,
    }
    if len(numeric_cols) < 2:
        return result

    corr = df[numeric_cols].corr(method="pearson", numeric_only=True)
    # Заполняем диагональ NaN, чтобы не учитывать r(x,x)=1 в максимуме.
    matrix_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(matrix_arr, np.nan)
    abs_arr = np.abs(matrix_arr)
    max_abs = float(np.nanmax(abs_arr)) if np.isfinite(abs_arr).any() else None

    result["matrix"] = {
        col: {other: float(corr.loc[col, other]) for other in numeric_cols}
        for col in numeric_cols
    }
    result["max_abs_correlation"] = max_abs

    if target_col and target_col in numeric_cols:
        target_corrs = corr[target_col].drop(target_col).abs()
        if not target_corrs.empty:
            result["target_correlation_max"] = float(target_corrs.max())
            result["target_correlation_by_column"] = {
                col: float(val) for col, val in target_corrs.items()
            }
    return result


# =============================================================================
#                       8. РАСПРЕДЕЛЕНИЯ ДЛЯ UI
# =============================================================================


def compute_numeric_distribution(values: pd.Series, bins: int = 30) -> dict[str, Any]:
    """
    Гистограмма для отрисовки на фронте через Plotly.

    Возвращает {bin_edges, counts}. NaN-значения отбрасываются. Если
    в колонке нет конечных значений — возвращается пустая гистограмма.
    """
    clean = values.dropna().to_numpy(dtype=float, na_value=np.nan)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(clean, bins=bins)
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}


def compute_categorical_distribution(
    values: pd.Series, top: int = 20
) -> dict[str, Any]:
    """
    Bar chart для категориальной колонки: топ-`top` значений и их частоты.
    Остальные категории агрегируются в «Прочее».
    """
    clean = values.dropna()
    if clean.empty:
        return {"categories": [], "counts": [], "other_count": 0}
    counts = clean.value_counts()
    head = counts.head(top)
    other = int(counts.iloc[top:].sum()) if len(counts) > top else 0
    return {
        "categories": [str(k) for k in head.index],
        "counts": [int(v) for v in head.values],
        "other_count": other,
    }


def compute_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Распределения для всех колонок (для отрисовки на фронте)."""
    return {
        "numeric": {col: compute_numeric_distribution(df[col]) for col in numeric_cols},
        "categorical": {
            col: compute_categorical_distribution(df[col]) for col in categorical_cols
        },
    }


# =============================================================================
#                       9. ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================


def compute_meta_features(
    df: pd.DataFrame, target_col: str | None = None
) -> dict[str, Any]:
    """
    Главная точка входа профайлера: вычисление полного набора meta-features.

    Включает:
    - Структурные характеристики (размеры, типы, объём в памяти).
    - Пропуски (общая доля и по столбцам).
    - Дубликаты строк.
    - Статистики по числовым колонкам (моменты, выбросы, нормальность).
    - Статистики по категориальным колонкам (кардинальность, энтропия).
    - Целевую переменную (тип, дисбаланс, MI признаков с target).
    - Корреляционную матрицу и связь признаков с числовым target.
    - Распределения для отрисовки графиков на фронте.
    - Информацию о сэмплировании (если применялось).

    Все рандомизированные операции (сэмплирование, MI) используют
    фиксированный RANDOM_STATE — требование воспроизводимости из ТЗ.

    Args:
        df: исходный DataFrame.
        target_col: имя целевого столбца (None для задач кластеризации).

    Returns:
        Плоский словарь meta-features с вложенными структурами для UI
        (distributions, correlations, *_by_column).
    """
    sampled_df, sampling_info = maybe_sample(df, target_col)
    columns = categorize_columns(sampled_df)
    numeric_cols = columns["numeric"]
    categorical_cols = columns["categorical"]

    # Целевая переменная исключается из общих числовых статистик: для неё
    # есть отдельные target_*-поля и MI. Иначе target смешается с features.
    feature_numeric = [c for c in numeric_cols if c != target_col]
    feature_categorical = [c for c in categorical_cols if c != target_col]

    structural = compute_structural_features(sampled_df)
    missing = compute_missing_features(sampled_df)
    duplicates = compute_duplicate_features(sampled_df)
    numeric = compute_numeric_features(sampled_df, feature_numeric)
    categorical = compute_categorical_features(sampled_df, feature_categorical)
    target = compute_target_features(sampled_df, target_col)
    mi = compute_mi_with_target(sampled_df, target_col)
    correlations = compute_correlations(sampled_df, numeric_cols, target_col)
    distributions = compute_distributions(sampled_df, numeric_cols, categorical_cols)

    # Плоский словарь со скалярами (для embedding в Спринте 3) + вложенные
    # детали для UI и проверок качества.
    meta: dict[str, Any] = {
        **structural,
        **missing,
        **duplicates,
        **numeric,
        **categorical,
        **target,
        "max_abs_correlation": correlations["max_abs_correlation"],
        "target_correlation_max": correlations["target_correlation_max"],
        "target_mutual_information_max": mi["max"] if mi else None,
        "target_mutual_information_mean": mi["mean"] if mi else None,
        "target_mutual_information_by_column": mi["by_column"] if mi else None,
        "target_correlation_by_column": correlations["target_correlation_by_column"],
        "correlation_matrix": correlations["matrix"],
        "distributions": distributions,
        "sampling": sampling_info,
        "target_column": target_col,
    }
    return meta
