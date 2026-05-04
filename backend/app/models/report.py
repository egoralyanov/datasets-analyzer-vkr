"""
ORM-модель PDF-отчёта по результатам анализа датасета.

Соответствует таблице `reports` (см. .knowledge/architecture/database.md).
Запись отчёта создаётся при нажатии кнопки «Сгенерировать отчёт» на странице
анализа: фон-задача рендерит HTML через Jinja2 + WeasyPrint, складывает PDF
на диск в `/data/reports/{user_id}/{report_id}.pdf` и обновляет запись
финальным статусом.

Состояния `status`:
- `pending`  — запись создана, фоновая задача ещё не стартовала.
- `running`  — фоновая задача рендерит PDF.
- `success`  — PDF успешно сгенерирован, лежит по `file_path`.
- `failed`   — рендер упал, в `error` лежит первые 1000 символов исключения.

`user_id` денормализован (формально достаточно `analysis.user_id`), чтобы
проверка владения отчётом и фильтрация по пользователю не требовали JOIN-а
с `analyses`. Композитный индекс `(user_id, status)` ускоряет запросы
«мои pending/running отчёты» в админке и при защите от двойной генерации.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.user import User


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_user_status", "user_id", "status"),
    )

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
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="reports")
    user: Mapped["User"] = relationship(back_populates="reports")
