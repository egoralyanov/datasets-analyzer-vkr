"""
Сервис подбора похожих датасетов через pgvector.

Состав:
- `meta_features_to_embedding(meta, scaler)` — векторизация meta-features
  пользовательского датасета в pgvector-совместимый embedding длины 128.
- `find_similar_datasets(db, query_embedding, ...)` — поиск top-K соседей
  в каталоге `external_datasets` через pgvector-операторы.
- `_load_scaler_safe()` — ленивая загрузка обученного StandardScaler с
  graceful degradation при отсутствии файла.

Алгоритм:
1. Сборка вектора. `meta_features_to_vector(meta)` возвращает 16 признаков в
   каноническом порядке (см. `ml/feature_vector.py`). Применяем
   `scaler.transform([vector])` — тот же scaler.pkl, на котором обучался
   мета-классификатор. На выходе — нормализованный вектор той же длины 16.
2. Padding до 128. Размерность pgvector-колонки `vector(128)` зашита в схеме
   (см. `ml/feature_vector.py:EMBEDDING_DIM`). Заполняем нулями: первые 16
   позиций — отскейленный вектор, остальные 112 — нули. Это формальный padding
   ради стабильной размерности; никакой семантики у «нулевых хвостов» нет.
3. Поиск. SQL вида `ORDER BY embedding <=> :query LIMIT k`. Оператор выбирается
   по словарю `OPERATORS` с whitelist-проверкой — оператор нельзя
   параметризовать через `:param`, его подставляем в f-string, но только из
   фиксированного множества (защита от SQL-инъекции).

См. `.knowledge/methods/dataset-matching.md` (полная теория и обоснование cosine
как основной метрики) и `.knowledge/architecture/database.md` (HNSW-индекс
`external_datasets_embedding_idx` создан в Phase 1 миграции).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.models.external_dataset import ExternalDataset
from app.repositories import external_dataset_repo
from ml.feature_vector import EMBEDDING_DIM, meta_features_to_vector


logger = logging.getLogger(__name__)


# Путь к обученному StandardScaler. Артефакт `scaler.pkl` создаётся скриптом
# `ml/train_meta_classifier.py` и не коммитится — переобучается через
# `make seed-all` (привязка к версии sklearn).
SCALER_PATH = Path("ml/models/scaler.pkl")


# Whitelist операторов pgvector. Маппинг используется и в find_similar_datasets,
# и в репозитории external_dataset_repo.find_similar — единая точка валидации.
# По умолчанию используется cosine; euclidean/manhattan нужны для
# исследовательской части РПЗ (сравнение метрик precision@K).
OPERATORS: dict[str, str] = {
    "cosine": "<=>",
    "euclidean": "<->",
    "manhattan": "<+>",
}


# Module-level кеш scaler'а, загружается лениво. Force-reload используется
# в тестах для monkeypatch'а пути к scaler.pkl. Кеш отдельный от
# task_recommender._SCALER — два сервиса не должны зависеть друг от друга.
_SCALER: Any | None = None
_LOADED: bool = False


def _load_scaler_safe(*, force_reload: bool = False) -> Any | None:
    """
    Безопасная загрузка StandardScaler из `backend/ml/models/scaler.pkl`.

    При отсутствии файла возвращает `None` и пишет warning. В analysis_service
    этот сценарий означает «embedding пользовательского датасета не считаем —
    остаётся NULL». Анализ при этом не валится.

    Args:
        force_reload: при True игнорируется кеш и файл перечитывается.
            Нужно тестам, чтобы monkeypatch'нуть путь.

    Returns:
        Объект `StandardScaler` либо `None`.
    """
    global _SCALER, _LOADED
    if _LOADED and not force_reload:
        return _SCALER

    try:
        if not SCALER_PATH.exists():
            logger.warning(
                "Scaler not found at %s — falling back to no-embedding mode. "
                "Run `make train-meta` to train the model first.",
                SCALER_PATH,
            )
            _SCALER = None
        else:
            _SCALER = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)
    except Exception:  # noqa: BLE001 — любой сбой загрузки = graceful degradation
        logger.exception("Failed to load scaler; falling back to no-embedding mode")
        _SCALER = None

    _LOADED = True
    return _SCALER


def meta_features_to_embedding(
    meta: dict[str, Any],
    scaler: Any,
) -> list[float]:
    """
    Векторизация meta-features в pgvector-совместимый embedding длины 128.

    Возвращает список Python-float'ов (а не np.ndarray), потому что
    pgvector-sqlalchemy с list работает стабильнее на разных версиях
    (см. `.knowledge/architecture/database.md` — заметка про сериализацию).

    Args:
        meta: словарь meta-features из `compute_meta_features()`.
        scaler: обученный sklearn `StandardScaler` из `_load_scaler_safe()`.
            Должен быть not None — проверка лежит на вызывающей стороне.

    Returns:
        Список из `EMBEDDING_DIM` (128) чисел: первые 16 — отскейленный
        вектор канонических признаков, остальные 112 — нули (padding).
    """
    raw_vector = meta_features_to_vector(meta).reshape(1, -1)
    scaled = scaler.transform(raw_vector)[0]
    embedding = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    embedding[: scaled.shape[0]] = scaled
    return embedding.tolist()


def find_similar_datasets(
    db: Session,
    query_embedding: list[float],
    *,
    task_type_filter: str | None = None,
    top_k: int = 5,
    metric: str = "cosine",
) -> list[ExternalDataset]:
    """
    Поиск top-K похожих датасетов через pgvector.

    Args:
        db: SQLAlchemy-сессия.
        query_embedding: вектор пользовательского датасета (длина 128, list[float]).
        task_type_filter: опциональный фильтр по `task_type_code`. None — без фильтра.
        top_k: число записей в результате (по умолчанию 5).
        metric: одна из `OPERATORS` (cosine / euclidean / manhattan). По умолчанию cosine.

    Returns:
        Список ORM-объектов `ExternalDataset`, отсортированных по distance
        возрастающе (ближайшие первыми). Distance прицеплен к каждой записи
        как атрибут `distance` (через `setattr` в репозитории).

    Raises:
        ValueError: при неизвестном `metric`. Это защита от SQL-инъекции —
        оператор подставляется в raw SQL через f-string, поэтому строгий
        whitelist обязателен.
    """
    if metric not in OPERATORS:
        raise ValueError(
            f"Unknown metric: {metric!r}. Allowed: {sorted(OPERATORS.keys())}"
        )
    operator = OPERATORS[metric]
    return external_dataset_repo.find_similar(
        db,
        query_embedding=query_embedding,
        task_type_filter=task_type_filter,
        top_k=top_k,
        operator=operator,
    )
