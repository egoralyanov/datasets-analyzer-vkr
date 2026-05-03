"""
Unit-тесты для сервиса подбора похожих датасетов через pgvector.

Тесты опираются на session-scoped фикстуру `_ensure_external_datasets_seeded`
из conftest.py, которая один раз заливает 29 записей из real_set.json в БД.
TRUNCATE users CASCADE между тестами не трогает external_datasets (нет FK),
так что каталог стабилен для всех тестов.

Покрытие согласно плану Phase 4 (5 тестов):
- размер embedding'а 128 (16 признаков + 112 нулей padding)
- поиск возвращает каталожный датасет как первого соседа сам себе
- top_k=5 → ровно 5 записей
- task_type_filter сужает выдачу
- is_active=False исключает запись из top-K
"""
from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.models.external_dataset import ExternalDataset
from app.services.dataset_matcher import (
    _load_scaler_safe,
    find_similar_datasets,
    meta_features_to_embedding,
)
from ml.feature_vector import EMBEDDING_DIM


@pytest.fixture
def matcher_db() -> Iterator[Session]:
    """
    Сессия БД для тестов matcher'а (без TRUNCATE-логики из db_session).

    db_session из conftest вызывает TRUNCATE users CASCADE в teardown, который
    нам не нужен и потенциально тормозит. external_datasets не зависят от
    пользователей, поэтому открываем чистую сессию.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _scaler_or_skip():
    scaler = _load_scaler_safe(force_reload=True)
    if scaler is None:
        pytest.skip("scaler.pkl not found — run `make train-meta` first")
    return scaler


def _catalog_record(matcher_db: Session) -> ExternalDataset:
    """Любая каталожная запись для тестов similarity."""
    record = matcher_db.execute(
        select(ExternalDataset).where(ExternalDataset.is_active.is_(True)).limit(1)
    ).scalar_one_or_none()
    if record is None:
        pytest.skip("Catalog is empty — run `make seed-catalog` first")
    return record


def test_meta_features_to_embedding_size_128():
    """
    Embedding имеет ровно 128 элементов: первые 16 — отскейленный вектор
    канонических признаков, остальные 112 — padding нулями.
    """
    scaler = _scaler_or_skip()
    fake_meta = {"n_rows": 1000, "n_cols": 10, "memory_mb": 0.5}
    embedding = meta_features_to_embedding(fake_meta, scaler)
    assert len(embedding) == EMBEDDING_DIM == 128
    # Первые 16 элементов — результат scaler.transform, не all-zeros в общем случае.
    head = embedding[:16]
    assert any(abs(v) > 1e-9 for v in head), "первые 16 не должны быть all-zeros"
    # Хвост — нули padding'а.
    tail = embedding[16:]
    assert all(v == 0.0 for v in tail), "padding должен быть нулевым"


def test_dataset_similar_to_itself(matcher_db: Session):
    """
    Поиск top_k=1 по embedding'у каталожного датасета должен вернуть его же
    с distance ≈ 0 (cosine distance к самому себе = 0).
    """
    record = _catalog_record(matcher_db)
    own_embedding = list(record.embedding) if record.embedding is not None else []
    assert own_embedding, "у каталожной записи должен быть embedding"

    results = find_similar_datasets(
        matcher_db, own_embedding, top_k=1, metric="cosine"
    )
    assert len(results) == 1
    assert results[0].title == record.title
    assert results[0].distance < 1e-6


def test_find_similar_returns_topk(matcher_db: Session):
    """`top_k=5` возвращает ровно 5 записей при каталоге ≥ 5."""
    record = _catalog_record(matcher_db)
    own_embedding = list(record.embedding)
    results = find_similar_datasets(matcher_db, own_embedding, top_k=5)
    assert len(results) == 5
    # distance должен монотонно расти (или хотя бы не убывать).
    distances = [r.distance for r in results]
    assert distances == sorted(distances)


def test_find_similar_filters_by_task_type(matcher_db: Session):
    """`task_type_filter='REGRESSION'` возвращает только регрессионные."""
    record = _catalog_record(matcher_db)
    own_embedding = list(record.embedding)
    results = find_similar_datasets(
        matcher_db,
        own_embedding,
        top_k=10,
        task_type_filter="REGRESSION",
    )
    assert len(results) >= 1
    assert all(r.task_type_code == "REGRESSION" for r in results)


def test_inactive_datasets_not_returned(matcher_db: Session):
    """
    Запись с is_active=False не попадает в top-K.

    Меняем is_active на одной записи внутри теста и в finally возвращаем True,
    чтобы не отравлять каталог для последующих тестов в том же прогоне.
    """
    record = matcher_db.execute(
        select(ExternalDataset).where(ExternalDataset.is_active.is_(True)).limit(1)
    ).scalar_one()
    target_title = record.title
    own_embedding = list(record.embedding)

    record.is_active = False
    matcher_db.commit()
    try:
        results = find_similar_datasets(
            matcher_db, own_embedding, top_k=10, metric="cosine"
        )
        titles = {r.title for r in results}
        assert target_title not in titles, (
            f"inactive {target_title!r} не должен быть в top-K"
        )
    finally:
        record.is_active = True
        matcher_db.commit()
