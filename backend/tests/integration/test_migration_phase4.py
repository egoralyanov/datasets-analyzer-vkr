"""
Smoke-тесты на миграции Спринта 6, Phase 4.

⚠️ Помечены маркером `migration` — по умолчанию pytest их пропускает
(`addopts = "-m 'not migration'"` в pyproject.toml). Запуск отдельным
таргетом `make test-migrations`.

Тесты переключают схему БД через `alembic.command.downgrade/upgrade` и
могут влиять на состояние других тестов, если бы запускались в одном
проходе. Поэтому разнесены в отдельный suite.
"""
from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.core.db import engine

PHASE_4_1_REVISION = "d75615d1ab2f"
PREVIOUS_REVISION = "c15193c9b987"


def _alembic_config() -> Config:
    """alembic.ini лежит в /app внутри контейнера; cwd при запуске тестов = /app."""
    return Config("alembic.ini")


@pytest.mark.migration
def test_phase4_1_migration_roundtrip() -> None:
    """
    После downgrade(-1) колонка file_hash и индекс уходят; после upgrade(head)
    возвращаются. По итогу схема снова в head — другие тесты этого suite
    (если бы запускались за этим) увидели бы консистентное состояние.
    """
    cfg = _alembic_config()
    inspector = inspect(engine)

    # Стартовое состояние — head (включает file_hash).
    cols_before = {c["name"] for c in inspector.get_columns("datasets")}
    indexes_before = {i["name"] for i in inspector.get_indexes("datasets")}
    assert "file_hash" in cols_before, "Pre-condition: миграция должна быть применена до старта теста"
    assert "ix_datasets_user_file_hash_unique" in indexes_before

    # downgrade на одну ревизию назад — колонка и индекс должны исчезнуть.
    command.downgrade(cfg, "-1")
    inspector = inspect(engine)
    cols_after_down = {c["name"] for c in inspector.get_columns("datasets")}
    indexes_after_down = {i["name"] for i in inspector.get_indexes("datasets")}
    assert "file_hash" not in cols_after_down
    assert "ix_datasets_user_file_hash_unique" not in indexes_after_down

    # upgrade обратно — всё на месте.
    command.upgrade(cfg, "head")
    inspector = inspect(engine)
    cols_final = {c["name"] for c in inspector.get_columns("datasets")}
    indexes_final = {i["name"] for i in inspector.get_indexes("datasets")}
    assert "file_hash" in cols_final
    assert "ix_datasets_user_file_hash_unique" in indexes_final

    # Индекс действительно UNIQUE (это и есть основной инвариант для
    # дедупликации).
    target_idx = next(
        i
        for i in inspector.get_indexes("datasets")
        if i["name"] == "ix_datasets_user_file_hash_unique"
    )
    assert target_idx["unique"] is True
    assert target_idx["column_names"] == ["user_id", "file_hash"]
