"""
Сбор синтетической части обучающей выборки мета-классификатора.

Генерирует 150 датасетов через `sklearn.datasets.make_classification` и
`make_regression` с заранее заданной сеткой параметров и фиксированным
`random_state=42`. Для каждого датасета прогоняет полный профайлер из
Спринта 2 и сохраняет результат в `backend/ml/data/synthetic_set.json`.

Зачем синтетика. На 30 реальных датасетах мета-классификатор переобучается
(LeaveOneOut CV даёт огромную дисперсию, RF фактически запоминает выборку).
Синтетическая аугментация — стандартная техника meta-learning при малой
выборке реальных данных (Vanschoren 2018, arxiv.org/abs/1810.03548).
150 синтетических датасетов с разнообразными `n_samples`, `n_features`,
`n_classes`, `class_sep` / `noise` расширяют пространство признаков и
стабилизируют CV-оценку.

В каталог `external_datasets` синтетика **не идёт** — у неё нет осмысленного
title/source_url для UI. Этот файл служит только обучению Слоя 2.

Запуск: `make build-synthetic-set` (см. Makefile).
"""
from __future__ import annotations

import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from app.services import profiler
from ml.build_real_set import _json_default  # переиспользуем numpy-сериализатор


logger = logging.getLogger("build_synthetic_set")


OUTPUT_PATH = Path("ml/data/synthetic_set.json")
RANDOM_STATE = 42

# Сколько записей каждого типа генерируем.
N_CLASSIFICATION = 75
N_REGRESSION = 75


# =============================================================================
#                       1. СЕТКА ПАРАМЕТРОВ
# =============================================================================


def _classification_grid() -> list[dict[str, Any]]:
    """
    Полная декартова сетка параметров для make_classification:
    n_samples × n_features × n_classes × class_sep = 5×3×3×3 = 135 комбинаций.

    Из 135 берём 75 случайных c фиксированным RandomState — детерминированно
    (тот же random_state даёт тот же выбор).
    """
    grid = list(
        product(
            (200, 500, 1000, 2000, 5000),    # n_samples
            (5, 10, 20),                      # n_features
            (2, 3, 5),                        # n_classes
            (0.5, 1.0, 1.5),                  # class_sep
        )
    )
    rng = np.random.RandomState(RANDOM_STATE)
    chosen_idx = rng.choice(len(grid), size=N_CLASSIFICATION, replace=False)
    return [
        {
            "n_samples": grid[i][0],
            "n_features": grid[i][1],
            "n_classes": grid[i][2],
            "class_sep": grid[i][3],
        }
        for i in chosen_idx
    ]


def _regression_grid() -> list[dict[str, Any]]:
    """
    Декартова сетка для make_regression:
    n_samples × n_features × noise = 5×3×3 = 45 комбинаций.

    Нужно 75 датасетов — 45 < 75, поэтому делаем sample with replacement,
    но дополнительно варьируем `random_state` итерации, чтобы пары (params,
    seed) были уникальны и датасеты получались разные.
    """
    base = list(
        product(
            (200, 500, 1000, 2000, 5000),    # n_samples
            (5, 10, 20),                      # n_features
            (0.0, 0.1, 0.5),                  # noise
        )
    )
    rng = np.random.RandomState(RANDOM_STATE + 1)  # +1 чтобы не пересечься с classification rng
    chosen_idx = rng.choice(len(base), size=N_REGRESSION, replace=True)
    grid = []
    for i, idx in enumerate(chosen_idx):
        grid.append(
            {
                "n_samples": base[idx][0],
                "n_features": base[idx][1],
                "noise": base[idx][2],
                "random_state": RANDOM_STATE + i,  # уникальный seed на каждую запись
            }
        )
    return grid


# =============================================================================
#                       2. ГЕНЕРАЦИЯ ОДНОГО ДАТАСЕТА
# =============================================================================


def _generate_classification_record(idx: int, params: dict[str, Any]) -> dict[str, Any]:
    """make_classification → DataFrame → meta_features → запись JSON."""
    n_classes = params["n_classes"]
    n_features = params["n_features"]

    # n_informative должен быть хотя бы log2(n_classes * 2). Для n_classes=5
    # минимум log2(10) ≈ 3.32 → 4. Для безопасности возьмём max(2, log2(n_classes)+1).
    # Иначе make_classification кидает ValueError при больших n_classes / малых n_features.
    n_informative = max(2, int(np.ceil(np.log2(max(n_classes, 2)))) + 1)
    n_informative = min(n_informative, n_features)

    X, y = make_classification(
        n_samples=params["n_samples"],
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, min(2, n_features - n_informative)),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=params["class_sep"],
        random_state=RANDOM_STATE + idx,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    task_type_code = (
        "BINARY_CLASSIFICATION" if n_classes == 2 else "MULTICLASS_CLASSIFICATION"
    )
    meta = profiler.compute_meta_features(df, "target")
    return {
        "identifier": f"synthetic_classif_{idx:03d}",
        "source": "synthetic",
        "task_type_code": task_type_code,
        "target_column": "target",
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "params": params,
        "meta_features": meta,
    }


def _generate_regression_record(idx: int, params: dict[str, Any]) -> dict[str, Any]:
    """make_regression → DataFrame → meta_features → запись JSON."""
    n_features = params["n_features"]
    n_informative = max(1, n_features // 2)

    X, y = make_regression(
        n_samples=params["n_samples"],
        n_features=n_features,
        n_informative=n_informative,
        noise=params["noise"],
        random_state=params["random_state"],
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    meta = profiler.compute_meta_features(df, "target")
    return {
        "identifier": f"synthetic_regr_{idx:03d}",
        "source": "synthetic",
        "task_type_code": "REGRESSION",
        "target_column": "target",
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "params": params,
        "meta_features": meta,
    }


# =============================================================================
#                       3. ОСНОВНОЙ ПАЙПЛАЙН
# =============================================================================


def build_synthetic_set(output_path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    """Генерирует 75 классификационных + 75 регрессионных записей."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for i, params in enumerate(_classification_grid()):
        try:
            records.append(_generate_classification_record(i, params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FAIL classif #%d] %s: %s", i, params, exc)

    for i, params in enumerate(_regression_grid()):
        try:
            records.append(_generate_regression_record(i, params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FAIL regr #%d] %s: %s", i, params, exc)

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    by_task: dict[str, int] = {}
    for r in records:
        by_task[r["task_type_code"]] = by_task.get(r["task_type_code"], 0) + 1

    logger.info("=" * 60)
    logger.info("synthetic_set.json: %d записей", len(records))
    for code, count in sorted(by_task.items()):
        logger.info("  %s: %d", code, count)
    logger.info("=" * 60)

    return records


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    build_synthetic_set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
