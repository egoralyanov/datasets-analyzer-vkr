"""
Seed-скрипт каталога внешних датасетов в БД.

Читает `backend/ml/data/real_set.json` (создаётся `make build-real-set` в Phase 2),
для каждой записи считает embedding через текущий `scaler.pkl` и делает upsert
в таблицу `external_datasets` через `INSERT ... ON CONFLICT (title) DO UPDATE`.

Идемпотентен:
- При повторном запуске неизменные записи перезатираются теми же значениями
  (updated_at обновится). Это нормально для seed-скрипта.
- При обновлении meta-features (например, после правки профайлера) или scaler'а —
  embedding пересчитается, остальные иммутабельные поля сохранятся.

Если scaler.pkl отсутствует — скрипт завершается с ненулевым кодом и подсказкой
запустить `make train-meta`. Без scaler'а embedding не вычисляется.

Запуск:
    docker compose exec backend python -m seeds.seed_external_datasets
или через Makefile:
    make seed-catalog
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app.core.db import SessionLocal
from app.repositories.external_dataset_repo import upsert_external_dataset
from app.services.dataset_matcher import _load_scaler_safe, meta_features_to_embedding


logger = logging.getLogger("seed_external_datasets")


REAL_SET_PATH = Path("ml/data/real_set.json")


def seed() -> int:
    """
    Заливает каталог в БД. Возвращает число обработанных записей.

    Если scaler.pkl отсутствует — выбрасывает RuntimeError, чтобы main()
    мог завершиться с ненулевым кодом и сообщением.
    """
    if not REAL_SET_PATH.exists():
        raise RuntimeError(
            f"{REAL_SET_PATH} не найден. Запустите `make build-real-set` "
            "перед сидингом каталога."
        )

    scaler = _load_scaler_safe(force_reload=True)
    if scaler is None:
        raise RuntimeError(
            "scaler.pkl не найден. Запустите `make train-meta` — мета-классификатор "
            "и scaler обучаются вместе. Без scaler'а embedding посчитать нельзя."
        )

    records = json.loads(REAL_SET_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d records from %s", len(records), REAL_SET_PATH)

    with SessionLocal() as db:
        for r in records:
            embedding = meta_features_to_embedding(r["meta_features"], scaler)
            upsert_external_dataset(
                db,
                {
                    "title": r["title"],
                    "description": r["description"],
                    "source": r["source"],
                    "source_url": r["source_url"],
                    "task_type_code": r["task_type_code"],
                    "target_column": r["target_column"],
                    "n_rows": r["n_rows"],
                    "n_cols": r["n_cols"],
                    "tags": r["tags"],
                    "meta_features": r["meta_features"],
                    "embedding": embedding,
                    "is_active": True,
                },
            )
        db.commit()
    return len(records)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    try:
        n = seed()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"seed_external_datasets: upserted {n} records into external_datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
