"""Pydantic-схемы для датасетов."""
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    dtypes: dict[str, str]


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_size_bytes: int
    format: str
    n_rows: int | None
    n_cols: int | None
    uploaded_at: datetime


class DatasetWithPreview(DatasetResponse):
    preview: DatasetPreview
