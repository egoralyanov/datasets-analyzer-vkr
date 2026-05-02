"""
ORM-модель запуска анализа датасета.

Соответствует таблице `analyses` (см. .knowledge/architecture/database.md, раздел 3).
Хранит параметры запуска (target_column, hinted_task_type) и состояние выполнения.
Сами результаты профилирования живут в отдельной таблице `analysis_results`,
которая появится в Спринте 2.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.analysis_result import AnalysisResult
    from app.models.dataset import Dataset
    from app.models.quality_flag import QualityFlag
    from app.models.user import User


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_user_status", "user_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hinted_task_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="analyses")
    user: Mapped["User"] = relationship(back_populates="analyses")
    result: Mapped["AnalysisResult | None"] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )
    flags: Mapped[list["QualityFlag"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
