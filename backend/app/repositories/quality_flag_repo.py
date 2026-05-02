"""
Репозиторий для работы с quality_flags.

Тонкий слой над SQLAlchemy: bulk-вставка флагов после quality_checker и
выборка флагов конкретного анализа с JOIN на quality_rules для получения
severity и человекочитаемого названия правила.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.quality_flag import QualityFlag
from app.models.quality_rule import QualityRule


def bulk_create_flags(db: Session, flags: list[QualityFlag]) -> None:
    """
    Сохранение списка флагов одной транзакцией.

    Не делает commit — это ответственность вызывающего слоя (analysis_service):
    флаги создаются вместе с обновлением статуса анализа в одной транзакции,
    чтобы при падении частичных результатов не оставалось.
    """
    if not flags:
        return
    db.add_all(flags)
    db.flush()


def get_flags_for_analysis(
    db: Session, analysis_id: uuid.UUID
) -> list[tuple[QualityFlag, QualityRule]]:
    """
    Возвращает все флаги анализа с приклеенным правилом.

    JOIN на quality_rules даёт severity, code и name — ровно то, что
    нужно для отрисовки списка флагов на фронте без N+1 запросов.

    Args:
        db: SQLAlchemy-сессия.
        analysis_id: UUID анализа.

    Returns:
        Список кортежей (QualityFlag, QualityRule), отсортированных по
        severity (critical → warning → info) и времени создания.
    """
    # Severity упорядочивается через CASE: critical(0) → warning(1) → info(2),
    # чтобы критичные флаги всегда показывались первыми.
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    rows = (
        db.execute(
            select(QualityFlag, QualityRule)
            .join(QualityRule, QualityFlag.rule_id == QualityRule.id)
            .where(QualityFlag.analysis_id == analysis_id)
        )
        .all()
    )
    return sorted(
        [(flag, rule) for flag, rule in rows],
        key=lambda fr: (severity_order.get(fr[1].severity, 99), fr[0].created_at),
    )
