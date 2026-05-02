"""
Чтение CSV/XLSX-датасетов и формирование preview для UI.

Для CSV выполняется автоопределение кодировки (через chardet) и разделителя
(через csv.Sniffer) — типовая ситуация для русскоязычных файлов из Excel
в cp1251 с разделителем «;». Для XLSX используется openpyxl-движок pandas.
"""
import csv
from pathlib import Path
from typing import Any

import chardet
import pandas as pd

# Сколько байт читаем с начала CSV для детекции кодировки и разделителя.
_SNIFF_BYTES = 10_000
# Кандидаты в разделители для csv.Sniffer (запятая, ;, табуляция, |).
_DELIMITER_CANDIDATES = ",;\t|"
# Запасная кодировка, если chardet вернул неуверенный результат.
_FALLBACK_ENCODING = "utf-8"


def detect_encoding(path: Path) -> str:
    """
    Определяет кодировку файла по первым ~10 КБ.

    Если chardet возвращает None или малую уверенность — fallback на utf-8.
    Источник: .knowledge/troubleshooting.md, раздел про cp1251.
    """
    with path.open("rb") as f:
        raw = f.read(_SNIFF_BYTES)
    detected = chardet.detect(raw)
    encoding = detected.get("encoding")
    confidence = detected.get("confidence") or 0.0
    if not encoding or confidence < 0.5:
        return _FALLBACK_ENCODING
    return encoding


def _detect_csv_delimiter(path: Path, encoding: str) -> str:
    """Определяет разделитель CSV по первой строке через csv.Sniffer."""
    with path.open("r", encoding=encoding, errors="replace") as f:
        sample = f.read(_SNIFF_BYTES)
    try:
        return csv.Sniffer().sniff(sample, delimiters=_DELIMITER_CANDIDATES).delimiter
    except csv.Error:
        # Sniffer не справился (одна колонка, странные данные) — используем запятую.
        return ","


def _read_dataframe(path: Path, fmt: str) -> pd.DataFrame:
    """Загружает датасет в DataFrame с учётом формата."""
    if fmt == "csv":
        encoding = detect_encoding(path)
        delimiter = _detect_csv_delimiter(path, encoding)
        return pd.read_csv(path, encoding=encoding, sep=delimiter)
    if fmt == "xlsx":
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Неподдерживаемый формат: {fmt}")


def read_dataset_full(path: Path, fmt: str) -> pd.DataFrame:
    """
    Загружает датасет целиком в DataFrame.

    Используется анализом (профайлер + quality checker) — в отличие от
    read_dataset_preview, который отдаёт только первые 100 строк для UI.
    Для очень больших датасетов профайлер сам выполняет сэмплирование
    до 50 000 строк (см. profiler.maybe_sample).
    """
    return _read_dataframe(path, fmt)


def read_dataset_preview(path: Path, fmt: str, max_rows: int = 100) -> dict[str, Any]:
    """
    Читает датасет и возвращает структуру для UI: первые max_rows строк,
    список колонок и dtype'ы (как строки — JSON-сериализуемо).

    Args:
        path: путь к файлу на диске.
        fmt: 'csv' или 'xlsx'.
        max_rows: сколько строк отдать в preview (по умолчанию 100).

    Returns:
        dict со следующими ключами:
        - columns: list[str] — имена колонок,
        - rows: list[list[Any]] — первые max_rows строк (NaN заменены на None),
        - dtypes: dict[str, str] — тип каждой колонки в виде строки,
        - n_rows: общее число строк датасета,
        - n_cols: общее число колонок.
    """
    df = _read_dataframe(path, fmt)
    head = df.head(max_rows)
    # NaN не сериализуется в JSON напрямую — приводим к None через where(notna).
    rows_clean = head.where(head.notna(), None).values.tolist()
    return {
        "columns": [str(c) for c in df.columns],
        "rows": rows_clean,
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
