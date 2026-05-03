"""
Сбор реальной части обучающей выборки мета-классификатора.

Скрипт скачивает 30 эталонных датасетов из открытых источников
(`sklearn.datasets`, UCI ML Repository, GitHub raw), прогоняет каждый через
полный профайлер из Спринта 2 (`app.services.profiler.compute_meta_features`)
и сохраняет результат в `backend/ml/data/real_set.json`.

Этот файл служит **двумя одновременно** целями (см. `recommender-ml.md` и
`dataset-matching.md`):
1. Источник записей с `task_type_code in {BINARY, MULTICLASS, REGRESSION}`
   для обучения мета-классификатора Слоя 2 (CLUSTERING-записи отфильтровываются
   при сборке X/y в `train_meta_classifier.py`).
2. Источник записей **всех 4 типов** для каталога `external_datasets`
   (Phase 4 → `seed_external_datasets.py` читает этот же файл, считает
   embedding и грузит в БД).

Принципы:
- **Идемпотентность.** Повторный запуск не перескачивает уже обработанные
  датасеты: при совпадении `title` + `n_rows` + `n_cols` в существующем JSON
  запись пропускается. Чтобы пересчитать meta-features (например, после
  изменения профайлера) — удалить файл `real_set.json` и запустить заново.
- **Допустимая потеря.** Сетевые ошибки (UCI/GitHub лежат) логируются и
  обрабатываются `try/except`. План считает приемлемым 25-28 успешных
  записей из 30.
- **Полнота полей записи.** Сохраняем `title`, `description`, `source`,
  `source_url`, `task_type_code`, `target_column`, `n_rows`, `n_cols`,
  `tags`, `meta_features` — этого достаточно и для обучения, и для UI
  каталога (Phase 7 показывает карточку с `title` / `description` / `source` /
  `source_url`).

Запуск: `make build-real-set` (см. Makefile).
"""
from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import urllib.request

from app.services import profiler


logger = logging.getLogger("build_real_set")


# Путь к JSON-артефакту относительно репозитория. Скрипт запускается из
# /app, поэтому путь относительный к рабочей директории контейнера.
OUTPUT_PATH = Path("ml/data/real_set.json")


# Таймаут на скачивание UCI/GitHub: 30 секунд хватает большинству, и не блокирует
# скрипт надолго при сетевых проблемах.
HTTP_TIMEOUT_SEC = 30


# =============================================================================
#                       1. ОПИСАНИЕ ИСТОЧНИКОВ
# =============================================================================


@dataclass
class Source:
    """Описание одного источника датасета: где взять и как интерпретировать."""

    title: str
    description: str
    source: str  # "sklearn" | "uci" | "github"
    source_url: str | None
    task_type_code: str  # BINARY/MULTICLASS/REGRESSION/CLUSTERING
    target_column: str | None
    tags: list[str]
    loader: Callable[[], tuple[pd.DataFrame, str | None]]


def _make_sklearn_loader(
    name: str, target_col: str = "target"
) -> Callable[[], tuple[pd.DataFrame, str | None]]:
    """Создаёт ленивый loader для встроенного sklearn-датасета."""

    def load() -> tuple[pd.DataFrame, str | None]:
        from sklearn import datasets as sk_datasets

        loader_fn = getattr(sk_datasets, name)
        bunch = loader_fn(as_frame=True)
        df: pd.DataFrame = bunch.frame.copy()
        return df, target_col

    return load


def _make_sklearn_no_target_loader(
    name: str,
) -> Callable[[], tuple[pd.DataFrame, str | None]]:
    """Loader, который возвращает датасет без target — для CLUSTERING-каталога."""

    def load() -> tuple[pd.DataFrame, str | None]:
        from sklearn import datasets as sk_datasets

        loader_fn = getattr(sk_datasets, name)
        bunch = loader_fn(as_frame=True)
        df: pd.DataFrame = bunch.frame.copy()
        if "target" in df.columns:
            df = df.drop(columns=["target"])
        return df, None

    return load


def _http_get(url: str) -> bytes:
    """Скачивание URL с таймаутом. Простой urllib, без лишних зависимостей."""
    request = urllib.request.Request(url, headers={"User-Agent": "analyzer-vkr/1.0"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SEC) as response:
        return response.read()


def _make_csv_loader(
    url: str,
    target_col: str | None,
    *,
    sep: str = ",",
    header: int | None = 0,
    names: list[str] | None = None,
    na_values: list[str] | None = None,
) -> Callable[[], tuple[pd.DataFrame, str | None]]:
    """Loader для публичного CSV (GitHub raw, UCI direct)."""

    def load() -> tuple[pd.DataFrame, str | None]:
        raw = _http_get(url)
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=sep,
            header=header,
            names=names,
            na_values=na_values,
        )
        return df, target_col

    return load


# Имена колонок для нескольких UCI-датасетов без шапки.
_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

_HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]

_ABALONE_COLUMNS = [
    "sex", "length", "diameter", "height", "whole_weight",
    "shucked_weight", "viscera_weight", "shell_weight", "rings",
]

_GLASS_COLUMNS = [
    "id", "ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe", "type",
]


SOURCES: list[Source] = [
    # =========================================================================
    #                       BINARY CLASSIFICATION (10)
    # =========================================================================
    Source(
        title="Breast Cancer Wisconsin",
        description="Диагностика рака молочной железы по 30 числовым признакам биопсии.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="target",
        tags=["medical", "binary"],
        loader=_make_sklearn_loader("load_breast_cancer"),
    ),
    Source(
        title="Titanic — выживание пассажиров",
        description="Классический датасет для бинарной классификации на табличных смешанных данных.",
        source="github",
        source_url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="Survived",
        tags=["beginner", "binary", "tabular"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            target_col="Survived",
        ),
    ),
    Source(
        title="Pima Indians Diabetes",
        description="Прогноз диабета у индейцев пима по медицинским показателям.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="class",
        tags=["medical", "binary"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            target_col="class",
            header=None,
            names=[
                "pregnancies", "glucose", "blood_pressure", "skin_thickness",
                "insulin", "bmi", "diabetes_pedigree", "age", "class",
            ],
        ),
    ),
    Source(
        title="Ionosphere",
        description="Классификация радарных сигналов: «хороший»/«плохой» возврат от ионосферы.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="Class",
        tags=["physics", "binary"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv",
            target_col="Class",
            header=None,
            names=[f"f{i}" for i in range(34)] + ["Class"],
        ),
    ),
    Source(
        title="Sonar — Mines vs Rocks",
        description="Распознавание мин на основе спектра сонара (60 числовых признаков).",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="Class",
        tags=["acoustics", "binary"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv",
            target_col="Class",
            header=None,
            names=[f"f{i}" for i in range(60)] + ["Class"],
        ),
    ),
    Source(
        title="Banknote Authentication",
        description="Подлинность банкнот по статистическим признакам wavelet-преобразования.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="class",
        tags=["finance", "binary"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv",
            target_col="class",
            header=None,
            names=["variance", "skewness", "curtosis", "entropy", "class"],
        ),
    ),
    Source(
        title="Haberman Survival",
        description="Выживаемость пациенток через 5 лет после операции по поводу рака груди.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="survival",
        tags=["medical", "binary"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv",
            target_col="survival",
            header=None,
            names=["age", "operation_year", "axillary_nodes", "survival"],
        ),
    ),
    Source(
        title="Adult Income (Census)",
        description="Прогноз уровня дохода (>50K / ≤50K) по переписи 1994 года.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="income",
        tags=["census", "binary", "categorical_features"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            target_col="income",
            header=None,
            names=_ADULT_COLUMNS,
            na_values=[" ?"],
        ),
    ),
    Source(
        title="Spambase",
        description="Классификация писем как спам/не-спам по частотам слов и символов.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="is_spam",
        tags=["text", "binary"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
            target_col="is_spam",
            header=None,
            names=[f"f{i}" for i in range(57)] + ["is_spam"],
        ),
    ),
    Source(
        title="Heart Disease (Cleveland)",
        description="Бинаризированная диагностика сердечного заболевания по клиническим данным.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        task_type_code="BINARY_CLASSIFICATION",
        target_column="num",
        tags=["medical", "binary"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            target_col="num",
            header=None,
            names=_HEART_COLUMNS,
            na_values=["?"],
        ),
    ),

    # =========================================================================
    #                       MULTICLASS CLASSIFICATION (10)
    # =========================================================================
    Source(
        title="Iris — три вида ирисов",
        description="Учебный датасет для многоклассовой классификации по 4 морфологическим признакам.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="target",
        tags=["beginner", "multiclass"],
        loader=_make_sklearn_loader("load_iris"),
    ),
    Source(
        title="Wine Recognition",
        description="Классификация вин на 3 сорта по 13 химическим показателям.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="target",
        tags=["chemistry", "multiclass"],
        loader=_make_sklearn_loader("load_wine"),
    ),
    Source(
        title="Digits — рукописные цифры",
        description="Классификация цифр 0-9 по 8x8 пиксельным изображениям (64 признака).",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="target",
        tags=["multiclass", "image", "many_features"],
        loader=_make_sklearn_loader("load_digits"),
    ),
    Source(
        title="Wine Quality (Red)",
        description="Качество красного вина (3-8 баллов) по химическим показателям.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="quality",
        tags=["chemistry", "multiclass"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
            target_col="quality",
            header=None,
            names=[
                "fixed_acidity", "volatile_acidity", "citric_acid",
                "residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "density", "pH", "sulphates",
                "alcohol", "quality",
            ],
        ),
    ),
    Source(
        title="Wine Quality (White)",
        description="Качество белого вина (3-9 баллов) по химическим показателям.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="quality",
        tags=["chemistry", "multiclass"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv",
            target_col="quality",
            header=None,
            names=[
                "fixed_acidity", "volatile_acidity", "citric_acid",
                "residual_sugar", "chlorides", "free_sulfur_dioxide",
                "total_sulfur_dioxide", "density", "pH", "sulphates",
                "alcohol", "quality",
            ],
        ),
    ),
    Source(
        title="Glass Identification",
        description="Классификация стёкол по 9 химическим признакам (7 типов).",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="type",
        tags=["materials", "multiclass"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
            target_col="type",
            header=None,
            names=_GLASS_COLUMNS,
        ),
    ),
    Source(
        title="Iris (UCI mirror)",
        description="Зеркало Iris с UCI mirror на jbrownlee/Datasets, для проверки идемпотентности.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="species",
        tags=["beginner", "multiclass"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
            target_col="species",
            header=None,
            names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        ),
    ),
    Source(
        title="Ecoli — белковая локализация",
        description="Классификация E.coli белков по 8 признакам в 8 локализационных классов.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="localization",
        tags=["biology", "multiclass"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
            target_col="localization",
            sep=r"\s+",
            header=None,
            names=[
                "name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2",
                "localization",
            ],
        ),
    ),
    Source(
        title="Yeast — белковая локализация",
        description="Классификация дрожжевых белков по 8 биохимическим признакам в 10 классов.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="localization",
        tags=["biology", "multiclass"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
            target_col="localization",
            sep=r"\s+",
            header=None,
            names=[
                "name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac",
                "nuc", "localization",
            ],
        ),
    ),
    Source(
        title="Vehicle Silhouettes (Statlog)",
        description="Классификация силуэтов транспортных средств на 4 типа по 18 числовым признакам.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/vehicle.csv",
        task_type_code="MULTICLASS_CLASSIFICATION",
        target_column="class",
        tags=["automotive", "multiclass"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/vehicle.csv",
            target_col="class",
            header=None,
            names=[f"f{i}" for i in range(18)] + ["class"],
        ),
    ),

    # =========================================================================
    #                              REGRESSION (8)
    # =========================================================================
    Source(
        title="Diabetes — прогрессирование",
        description="Прогноз количественной меры прогрессирования диабета через год.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
        task_type_code="REGRESSION",
        target_column="target",
        tags=["medical", "regression"],
        loader=_make_sklearn_loader("load_diabetes"),
    ),
    Source(
        title="California Housing",
        description="Прогноз медианной стоимости жилья по районам Калифорнии.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        task_type_code="REGRESSION",
        target_column="MedHouseVal",
        tags=["regression", "real_estate"],
        loader=_make_sklearn_loader("fetch_california_housing", target_col="MedHouseVal"),
    ),
    Source(
        title="Boston Housing",
        description="Регрессия медианной стоимости жилья в Бостоне (классический датасет).",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv",
        task_type_code="REGRESSION",
        target_column="MEDV",
        tags=["regression", "real_estate"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv",
            target_col="MEDV",
            sep=r"\s+",
            header=None,
            names=[
                "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
                "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
            ],
        ),
    ),
    Source(
        title="Auto MPG",
        description="Расход топлива автомобиля (mpg) по 7 характеристикам.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
        task_type_code="REGRESSION",
        target_column="mpg",
        tags=["automotive", "regression"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
            target_col="mpg",
            sep=r"\s+",
            header=None,
            names=[
                "mpg", "cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year", "origin", "car_name",
            ],
            na_values=["?"],
        ),
    ),
    Source(
        title="Abalone — возраст моллюсков",
        description="Прогноз возраста моллюсков (число колец) по морфологическим измерениям.",
        source="github",
        source_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
        task_type_code="REGRESSION",
        target_column="rings",
        tags=["biology", "regression"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/abalone.csv",
            target_col="rings",
            header=None,
            names=_ABALONE_COLUMNS,
        ),
    ),
    Source(
        title="Yacht Hydrodynamics",
        description="Прогноз остаточного сопротивления яхты по 6 геометрическим параметрам корпуса.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        task_type_code="REGRESSION",
        target_column="resistance",
        tags=["fluid_dynamics", "regression"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            target_col="resistance",
            sep=r"\s+",
            header=None,
            names=[
                "longitudinal_position", "prismatic_coeff", "length_displacement",
                "beam_draught_ratio", "length_beam_ratio", "froude_number",
                "resistance",
            ],
        ),
    ),
    Source(
        title="Energy Efficiency (Heating Load)",
        description="Прогноз тепловой нагрузки здания по 8 архитектурным параметрам.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        task_type_code="REGRESSION",
        target_column="heating_load",
        tags=["energy", "regression"],
        loader=lambda: _load_energy_efficiency(),
    ),
    Source(
        title="Servo — время отклика",
        description="Время отклика сервомеханизма по 4 категориальным параметрам.",
        source="uci",
        source_url="https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data",
        task_type_code="REGRESSION",
        target_column="class",
        tags=["control", "regression"],
        loader=_make_csv_loader(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data",
            target_col="class",
            header=None,
            names=["motor", "screw", "pgain", "vgain", "class"],
        ),
    ),

    # =========================================================================
    #                            CLUSTERING (2)
    # =========================================================================
    Source(
        title="Mall Customers",
        description="Сегментация клиентов торгового центра по доходу и тратам (без меток).",
        source="github",
        source_url="https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv",
        task_type_code="CLUSTERING",
        target_column=None,
        tags=["clustering", "marketing"],
        loader=_make_csv_loader(
            "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv",
            target_col=None,
        ),
    ),
    Source(
        title="Iris (без меток)",
        description="Тот же Iris, но без целевой переменной — для демонстрации кластеризации.",
        source="sklearn",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
        task_type_code="CLUSTERING",
        target_column=None,
        tags=["clustering", "beginner"],
        loader=_make_sklearn_no_target_loader("load_iris"),
    ),
]


def _load_energy_efficiency() -> tuple[pd.DataFrame, str]:
    """
    Загрузка Energy Efficiency (xlsx). Отдельная функция, чтобы не тащить
    openpyxl в общий _make_csv_loader (xlsx не CSV).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    raw = _http_get(url)
    df = pd.read_excel(io.BytesIO(raw))
    df = df.rename(
        columns={
            "X1": "compactness", "X2": "surface_area", "X3": "wall_area",
            "X4": "roof_area", "X5": "overall_height", "X6": "orientation",
            "X7": "glazing_area", "X8": "glazing_distribution",
            "Y1": "heating_load", "Y2": "cooling_load",
        }
    )
    # Берём только heating_load как target, cooling_load выкидываем — иначе
    # будет регрессия с двумя targets, что наш профайлер не поддерживает.
    df = df.drop(columns=["cooling_load"])
    return df, "heating_load"


# =============================================================================
#                       2. ИДЕМПОТЕНТНОСТЬ
# =============================================================================


def _load_existing(path: Path) -> list[dict[str, Any]]:
    """Прочитать существующий JSON или вернуть пустой список."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Existing %s is not valid JSON, starting from scratch", path)
        return []


def _existing_signature(records: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    """Множество сигнатур (title, n_rows, n_cols) уже обработанных записей."""
    return {
        (r["title"], int(r.get("n_rows", -1)), int(r.get("n_cols", -1)))
        for r in records
    }


# =============================================================================
#                       3. ОСНОВНОЙ ПАЙПЛАЙН
# =============================================================================


def build_real_set(output_path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    """
    Главная точка входа. Возвращает финальный список записей.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_existing(output_path)
    seen = _existing_signature(records)

    succeeded = 0
    failed = 0
    skipped = 0

    for src in SOURCES:
        try:
            df, target_col = src.loader()
        except Exception as exc:  # noqa: BLE001 — план: ловить любые ошибки сети/парсинга
            logger.warning("[FAIL] %s: %s", src.title, exc)
            failed += 1
            continue

        n_rows = int(len(df))
        n_cols = int(df.shape[1])
        if (src.title, n_rows, n_cols) in seen:
            logger.info("[SKIP idempotent] %s (%d×%d)", src.title, n_rows, n_cols)
            skipped += 1
            continue

        try:
            meta = profiler.compute_meta_features(df, target_col)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FAIL profiler] %s: %s", src.title, exc)
            failed += 1
            continue

        records.append(
            {
                "title": src.title,
                "description": src.description,
                "source": src.source,
                "source_url": src.source_url,
                "task_type_code": src.task_type_code,
                "target_column": src.target_column,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "tags": src.tags,
                "meta_features": meta,
            }
        )
        seen.add((src.title, n_rows, n_cols))
        succeeded += 1
        logger.info(
            "[OK] %s (%d×%d, %s)",
            src.title, n_rows, n_cols, src.task_type_code,
        )

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    by_task: dict[str, int] = {}
    for r in records:
        by_task[r["task_type_code"]] = by_task.get(r["task_type_code"], 0) + 1

    logger.info("=" * 60)
    logger.info("real_set.json: %d записей всего", len(records))
    logger.info("Успешно добавлено: %d, пропущено (идемпотентность): %d, ошибок: %d",
                succeeded, skipped, failed)
    for code, count in sorted(by_task.items()):
        logger.info("  %s: %d", code, count)
    logger.info("=" * 60)

    return records


def _json_default(obj: Any) -> Any:
    """JSON-сериализатор для numpy-типов в meta_features."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    build_real_set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
