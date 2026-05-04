"""
CRUD-операции над таблицей reports.

Соответствует разделу архитектуры о PDF-отчётах. Все геттеры скоупятся по
`user_id`, чтобы не отдать чужой отчёт даже при подобранном UUID — паттерн
тот же, что в analysis_repo.

Race-condition note: проверка «нет ли уже pending/running отчёта для этого
analysis» через `get_active_report_for_analysis` НЕ защищена от гонки
двух одновременных POST'ов из браузера. Для ВКР с одним пользователем это
приемлемо — пользователь не нажимает кнопку дважды одновременно. Если
понадобится строгая защита — partial unique index `(analysis_id) WHERE
status IN ('pending','running')` закроет на уровне БД.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.report import Report


def create_report(
    db: Session,
    *,
    analysis_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Report:
    """Создаёт запись Report в статусе pending — БД-фиксация через commit."""
    report = Report(
        analysis_id=analysis_id,
        user_id=user_id,
        status="pending",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_report(
    db: Session, report_id: uuid.UUID, user_id: uuid.UUID
) -> Report | None:
    """Возвращает отчёт только если он принадлежит указанному пользователю.

    Скоуп по user_id обязательный — иначе подобранным UUID можно было бы
    дотянуться до чужого отчёта (и через download — до файла).
    """
    return db.scalar(
        select(Report).where(
            Report.id == report_id,
            Report.user_id == user_id,
        )
    )


def get_active_report_for_analysis(
    db: Session, analysis_id: uuid.UUID
) -> Report | None:
    """
    Возвращает последний pending/running отчёт по анализу (для проверки 409).

    Если такой есть — повторный POST не должен создавать второй отчёт,
    эндпоинт вернёт 409 + report_id существующего. Сортировка по
    created_at DESC — на случай маловероятной аномалии с двумя active
    записями выбираем самую свежую.
    """
    return db.scalar(
        select(Report)
        .where(
            Report.analysis_id == analysis_id,
            Report.status.in_(("pending", "running")),
        )
        .order_by(Report.created_at.desc())
    )
