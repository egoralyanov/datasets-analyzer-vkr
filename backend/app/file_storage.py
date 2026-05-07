"""Работа с локальным файловым хранилищем датасетов."""
import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import settings


# Размер буфера при стриминге файла для подсчёта SHA-256. 1 МБ — компромисс
# между числом read-сисколлов и пиковым потреблением памяти. Для типового
# CSV/XLSX порядка 10-50 МБ это 10-50 итераций.
_HASH_CHUNK_BYTES = 1024 * 1024


def compute_file_sha256(path: Path | str) -> str:
    """
    Считает SHA-256 содержимого файла, читая его чанками по 1 МБ.

    Используется при загрузке датасета (Спринт 6, Phase 4.1) для
    дедупликации: хэш сравнивается с существующими записями
    `datasets.file_hash` в рамках того же пользователя. Файл не
    загружается в память целиком — это важно для крупных Excel-файлов
    (до 100 МБ по лимиту MAX_FILE_SIZE_MB).

    Args:
        path: путь к файлу.

    Returns:
        hex-строка длиной 64 символа.
    """
    sha = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


def delete_report_file(relative_path: str) -> None:
    """
    Удаляет PDF-файл отчёта, разрешая путь относительно settings.REPORTS_DIR.

    `Report.file_path` хранится в БД как относительный путь
    `{user_id}/{report_id}.pdf` — абсолютный собирается через
    `REPORTS_DIR / file_path`. Используется при удалении анализа
    (Спринт 6, Phase 4.3) и пользователя (Phase 4.4) для зачистки orphan-PDF
    после cascade-удаления записей.
    """
    Path(settings.REPORTS_DIR, relative_path).unlink(missing_ok=True)
