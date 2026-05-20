"""CRUD-операции над таблицей datasets."""
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.dataset import Dataset
from app.models.report import Report


def create_dataset(
    db: Session,
    *,
    user_id: uuid.UUID,
    original_filename: str,
    storage_path: str,
    file_size_bytes: int,
    file_hash: str,
    fmt: str,
    n_rows: int | None,
    n_cols: int | None,
) -> Dataset:
    dataset = Dataset(
        user_id=user_id,
        original_filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=file_size_bytes,
        file_hash=file_hash,
        format=fmt,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_dataset(
    db: Session, dataset_id: uuid.UUID, user_id: uuid.UUID
) -> Dataset | None:
    """Возвращает датасет, только если он принадлежит указанному пользователю."""
    return db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == user_id,
        )
    )


def find_by_user_and_hash(
    db: Session, user_id: uuid.UUID, file_hash: str
) -> Dataset | None:
    """
    Ищет существующий датасет того же пользователя с указанным SHA-256.

    Используется при загрузке (Спринт 6, Phase 4.1) для дедупликации:
    при попытке загрузить файл с тем же содержимым возвращается 409 со
    ссылкой на существующий dataset_id, новая запись не создаётся.
    """
    return db.scalar(
        select(Dataset).where(
            Dataset.user_id == user_id,
            Dataset.file_hash == file_hash,
        )
    )


def get_dataset_any(db: Session, dataset_id: uuid.UUID) -> Dataset | None:
    """
    Возвращает датасет по ID без фильтра по пользователю.

    Используется в эндпоинтах, где доступ имеет admin (Спринт 6, Phase 4.2):
    `GET /api/datasets/{id}/usage` принимает запросы от владельца ИЛИ админа.
    Для не-админских вызовов следует использовать `get_dataset(db, id, user_id)`,
    который сразу скоупит по пользователю.
    """
    return db.get(Dataset, dataset_id)


def count_analyses_for_dataset(db: Session, dataset_id: uuid.UUID) -> int:
    """Число анализов датасета без фильтра по статусу (Спринт 6, Phase 4.2)."""
    result = db.scalar(
        select(func.count())
        .select_from(Analysis)
        .where(Analysis.dataset_id == dataset_id)
    )
    return int(result or 0)


def count_successful_reports_for_dataset(
    db: Session, dataset_id: uuid.UUID
) -> int:
    """
    Число успешно сгенерированных PDF-отчётов по всем анализам датасета.

    `status='success'` — только готовые артефакты (failed/pending для UI
    бесполезны). JOIN с analyses, потому что Report.dataset_id напрямую
    отсутствует — связь через analysis_id (см. модель Report).
    """
    result = db.scalar(
        select(func.count())
        .select_from(Report)
        .join(Analysis, Report.analysis_id == Analysis.id)
        .where(
            Analysis.dataset_id == dataset_id,
            Report.status == "success",
        )
    )
    return int(result or 0)


def list_datasets(db: Session, user_id: uuid.UUID) -> list[Dataset]:
    return list(
        db.scalars(
            select(Dataset)
            .where(Dataset.user_id == user_id)
            .order_by(Dataset.uploaded_at.desc())
        )
    )


def delete_dataset(db: Session, dataset: Dataset) -> None:
    db.delete(dataset)
    db.commit()
