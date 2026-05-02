"""
ORM-модель сработавшего флага качества для конкретного анализа.

Соответствует таблице `quality_flags` (см. .knowledge/architecture/database.md, раздел 6).
Каждый флаг привязан к анализу и к конкретному правилу из quality_rules. Поле `context`
(JSONB) содержит детали срабатывания: имя колонки, замеренное значение, превышенный порог.
Severity не дублируется в этой таблице — берётся через JOIN с quality_rules: если решим
повысить severity правила, исторические флаги автоматически переедут в новый уровень.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.quality_rule import QualityRule


class QualityFlag(Base):
    __tablename__ = "quality_flags"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quality_rules.id", ondelete="RESTRICT"),
        nullable=False,
    )
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="flags")
    rule: Mapped["QualityRule"] = relationship(back_populates="flags")
