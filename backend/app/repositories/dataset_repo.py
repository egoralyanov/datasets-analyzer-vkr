"""CRUD-операции над таблицей datasets."""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.dataset import Dataset


def create_dataset(
    db: Session,
    *,
    user_id: uuid.UUID,
    original_filename: str,
    storage_path: str,
    file_size_bytes: int,
    fmt: str,
    n_rows: int | None,
    n_cols: int | None,
) -> Dataset:
    dataset = Dataset(
        user_id=user_id,
        original_filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=file_size_bytes,
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
