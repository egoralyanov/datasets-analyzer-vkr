"""
API-эндпоинты для работы с датасетами.

См. .knowledge/architecture/api-contract.md, раздел 2.
"""
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.config import settings
from app.file_storage import delete_dataset_file, save_uploaded_file
from app.models.user import User
from app.repositories import dataset_repo
from app.schemas.dataset import DatasetPreview, DatasetResponse, DatasetWithPreview
from app.services.dataset_service import read_dataset_preview

router = APIRouter(prefix="/datasets", tags=["datasets"])

_ALLOWED_EXTENSIONS = {"csv", "xlsx"}


def _max_size_bytes() -> int:
    # Через функцию, чтобы monkeypatch settings.MAX_FILE_SIZE_MB в тестах работал.
    return int(settings.MAX_FILE_SIZE_MB * 1024 * 1024)


def _get_extension(filename: str | None) -> str | None:
    """Извлекает нижний регистр расширения. MIME намеренно не используем — его легко подделать."""
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[1].lower()


@router.post(
    "/upload",
    response_model=DatasetWithPreview,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    content_length: int | None = Header(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetWithPreview:
    ext = _get_extension(file.filename)
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Поддерживаются только файлы .csv и .xlsx",
        )

    # Раннее отсечение по Content-Length (страховка nginx-уровня).
    if content_length is not None and content_length > _max_size_bytes():
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Файл превышает лимит {settings.MAX_FILE_SIZE_MB} МБ",
        )

    storage_path, size = save_uploaded_file(file, current_user.id, ext)

    # Вторичная проверка по реальному размеру: header можно подделать.
    if size > _max_size_bytes():
        delete_dataset_file(storage_path)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Файл превышает лимит {settings.MAX_FILE_SIZE_MB} МБ",
        )

    try:
        preview = read_dataset_preview(Path(storage_path), ext)
    except Exception as e:
        delete_dataset_file(storage_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл: {e}",
        )

    dataset = dataset_repo.create_dataset(
        db,
        user_id=current_user.id,
        original_filename=file.filename or f"unnamed.{ext}",
        storage_path=storage_path,
        file_size_bytes=size,
        fmt=ext,
        n_rows=preview["n_rows"],
        n_cols=preview["n_cols"],
    )

    return DatasetWithPreview(
        **DatasetResponse.model_validate(dataset).model_dump(),
        preview=DatasetPreview(
            columns=preview["columns"],
            rows=preview["rows"],
            dtypes=preview["dtypes"],
        ),
    )


@router.get("", response_model=list[DatasetResponse])
def list_my_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DatasetResponse]:
    items = dataset_repo.list_datasets(db, current_user.id)
    return [DatasetResponse.model_validate(d) for d in items]


@router.get("/{dataset_id}", response_model=DatasetWithPreview)
def get_dataset_with_preview(
    dataset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetWithPreview:
    dataset = dataset_repo.get_dataset(db, dataset_id, current_user.id)
    if dataset is None:
        # 404 (не 403) — чтобы не палить факт существования чужого датасета.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Датасет не найден")

    try:
        preview = read_dataset_preview(Path(dataset.storage_path), dataset.format)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл: {e}",
        )

    return DatasetWithPreview(
        **DatasetResponse.model_validate(dataset).model_dump(),
        preview=DatasetPreview(
            columns=preview["columns"],
            rows=preview["rows"],
            dtypes=preview["dtypes"],
        ),
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_dataset(
    dataset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    dataset = dataset_repo.get_dataset(db, dataset_id, current_user.id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Датасет не найден")
    storage_path = dataset.storage_path
    dataset_repo.delete_dataset(db, dataset)
    delete_dataset_file(storage_path)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
