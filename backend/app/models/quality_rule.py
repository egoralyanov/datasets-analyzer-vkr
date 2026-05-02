"""
ORM-модель правила проверки качества данных.

Соответствует таблице `quality_rules` (см. .knowledge/architecture/database.md, раздел 5,
и .knowledge/methods/quality-checks.md — описание всех 12 правил).

Справочник правил живёт в БД, чтобы пороги можно было менять без редеплоя:
поле `thresholds` — JSONB вида `{"max_col_missing_pct": 0.3}`. Сами правила
кодируются в backend/app/services/quality_checker.py — там реализована логика
срабатывания, а в БД хранятся только метаданные (имя, описание, severity, пороги).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.quality_flag import QualityFlag


class QualityRule(Base):
    __tablename__ = "quality_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    flags: Mapped[list["QualityFlag"]] = relationship(back_populates="rule")
