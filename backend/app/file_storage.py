"""Работа с локальным файловым хранилищем датасетов."""
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


def _user_dir(user_id: uuid.UUID) -> Path:
    path = Path(settings.DATASETS_DIR) / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(
    file: UploadFile, user_id: uuid.UUID, ext: str
) -> tuple[str, int]:
    """
    Сохраняет загруженный файл под уникальным UUID-именем.

    Имя на диске никогда не совпадает с original_filename — это снимает
    риск коллизий и path-traversal через подделанное имя файла.
    Возвращает (storage_path, размер_в_байтах).
    """
    storage_uuid = uuid.uuid4()
    storage_path = _user_dir(user_id) / f"{storage_uuid}.{ext}"
    file.file.seek(0)
    with storage_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return str(storage_path), storage_path.stat().st_size


def delete_dataset_file(storage_path: str) -> None:
    """Удаляет файл с диска. Если файла уже нет — молча игнорируем."""
    Path(storage_path).unlink(missing_ok=True)
