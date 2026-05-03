"""
Репозиторий каталога внешних датасетов: upsert + поиск top-K через pgvector.

Содержит две публичные операции:

- `upsert_external_dataset(db, fields)` — идемпотентная вставка/обновление
  записи каталога. Ключ идемпотентности — `title` (UNIQUE в схеме).
  При конфликте обновляются ТОЛЬКО `embedding`, `meta_features`, `is_active`,
  `n_rows`, `n_cols` и `updated_at` — поля, которые могут законно меняться при
  пересборке каталога (например, после нового scaler.pkl или обновления
  профайлера). Иммутабельные поля — `title`, `description`, `source`,
  `source_url`, `task_type_code`, `target_column`, `tags` — при upsert не
  затирают существующие значения, чтобы администратор мог отредактировать
  описание через UI/SQL без отката seed-скриптом.

- `find_similar(db, query_embedding, ...)` — top-K соседей через pgvector.
  Оператор подставляется в SQL через f-string и проверяется на уровне сервиса
  (`dataset_matcher.OPERATORS`). На уровне репозитория повторно валидируем
  оператор через явный whitelist для defense-in-depth.

См. `.knowledge/methods/dataset-matching.md` (раздел «Реализация поиска») и
`.knowledge/architecture/database.md` (структура таблицы `external_datasets`,
HNSW-индекс на `embedding`).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.external_dataset import ExternalDataset


def _jsonb_safe(value: Any) -> Any:
    """
    Рекурсивно заменяет NaN/Infinity на None в произвольной dict/list-структуре.

    PostgreSQL JSONB не принимает NaN/Infinity (это нестандарт JSON), а профайлер
    может выдавать их в `correlation_matrix` для пар колонок без определённой
    корреляции (например, когда одна колонка константна — pandas возвращает
    NaN в r). Без очистки seed-скрипт падает с
    `psycopg.errors.InvalidTextRepresentation: Token "NaN" is invalid`.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _jsonb_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonb_safe(v) for v in value]
    return value


# Whitelist операторов pgvector — повторная защита перед raw SQL. Должен
# совпадать с `dataset_matcher.OPERATORS`; рассинхронизация → ValueError.
_ALLOWED_OPERATORS = frozenset({"<=>", "<->", "<+>"})


# Поля, которые разрешено обновлять при ON CONFLICT (title) DO UPDATE.
# Все остальные поля (description, source_url, source, task_type_code,
# target_column, tags) считаются иммутабельными после первичной заливки.
_UPSERT_UPDATE_FIELDS = (
    "embedding",
    "meta_features",
    "is_active",
    "n_rows",
    "n_cols",
)


def upsert_external_dataset(
    db: Session,
    fields: dict[str, Any],
) -> ExternalDataset:
    """
    Идемпотентная вставка записи в `external_datasets`.

    Использует PostgreSQL `INSERT ... ON CONFLICT (title) DO UPDATE`.
    При конфликте обновляются только мутабельные поля + `updated_at`.

    Args:
        db: SQLAlchemy-сессия. Commit вызывает вызывающая сторона.
        fields: словарь со всеми полями записи (title, description, source,
            source_url, task_type_code, target_column, n_rows, n_cols, tags,
            meta_features, embedding, is_active).

    Returns:
        ORM-объект `ExternalDataset`, перечитанный из БД после upsert'а.
    """
    now = datetime.now(timezone.utc)
    sanitized = {**fields}
    if "meta_features" in sanitized:
        sanitized["meta_features"] = _jsonb_safe(sanitized["meta_features"])
    if "tags" in sanitized:
        sanitized["tags"] = _jsonb_safe(sanitized["tags"])
    insert_values = {**sanitized, "created_at": now, "updated_at": now}

    set_on_conflict: dict[str, Any] = {
        f: insert_values[f] for f in _UPSERT_UPDATE_FIELDS if f in insert_values
    }
    set_on_conflict["updated_at"] = now

    stmt = (
        insert(ExternalDataset)
        .values(**insert_values)
        .on_conflict_do_update(
            index_elements=["title"],
            set_=set_on_conflict,
        )
    )
    db.execute(stmt)

    # Возвращаем уже перезаписанный объект — нужен для тестов/логов.
    return db.execute(
        select(ExternalDataset).where(ExternalDataset.title == fields["title"])
    ).scalar_one()


def find_similar(
    db: Session,
    *,
    query_embedding: list[float],
    task_type_filter: str | None,
    top_k: int,
    operator: str,
) -> list[ExternalDataset]:
    """
    Top-K соседей в каталоге через pgvector.

    SQL формируется через `text()` с явной подстановкой оператора в f-string —
    оператор pgvector нельзя параметризовать через `:param`. Защита от
    инъекции — whitelist `_ALLOWED_OPERATORS`.

    Args:
        db: SQLAlchemy-сессия.
        query_embedding: вектор длины 128.
        task_type_filter: опциональный фильтр по `task_type_code`.
        top_k: число записей в результате.
        operator: один из `<=>`, `<->`, `<+>`.

    Returns:
        Список `ExternalDataset`. К каждому объекту через `setattr` прицеплен
        атрибут `distance: float` — отсортированный distance от query.

    Raises:
        ValueError: при недопустимом операторе.
    """
    if operator not in _ALLOWED_OPERATORS:
        raise ValueError(
            f"Unknown pgvector operator: {operator!r}. "
            f"Allowed: {sorted(_ALLOWED_OPERATORS)}"
        )

    # pgvector принимает строку формата '[0.1, 0.2, ...]' для bind-параметра
    # с типом vector. Передача list[float] напрямую через bindparam в text()
    # не работает — нужен текстовый кастинг.
    embedding_literal = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"

    # Явные касты: pgvector принимает текст вида '[0.1,...]' через CAST AS vector;
    # task_type явно кастуется в TEXT, иначе Postgres не выводит тип параметра при NULL
    # (psycopg.errors.AmbiguousParameter).
    sql = f"""
        SELECT *,
               (embedding {operator} CAST(:emb AS vector)) AS distance
        FROM external_datasets
        WHERE is_active = TRUE
          AND (CAST(:task_type AS TEXT) IS NULL
               OR task_type_code = CAST(:task_type AS TEXT))
        ORDER BY embedding {operator} CAST(:emb AS vector)
        LIMIT :top_k
    """

    rows = db.execute(
        text(sql).bindparams(
            bindparam("emb", value=embedding_literal),
            bindparam("task_type", value=task_type_filter),
            bindparam("top_k", value=top_k),
        )
    ).all()

    # Превращаем Row в ORM ExternalDataset, прицепляя distance как атрибут.
    results: list[ExternalDataset] = []
    for row in rows:
        mapping = row._mapping
        obj = db.get(ExternalDataset, mapping["id"])
        if obj is None:
            continue  # запись могла быть удалена между SELECT и get — пропускаем
        # distance — внешнее поле, не часть ORM-модели; кладём как обычный атрибут.
        setattr(obj, "distance", float(mapping["distance"]))
        results.append(obj)
    return results
