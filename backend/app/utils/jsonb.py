"""
Утилиты для безопасной записи в PostgreSQL JSONB.

PostgreSQL JSONB не принимает NaN/Infinity (это нестандарт JSON), а
профайлер и обучение моделей могут их выдавать в нескольких местах:
- correlation_matrix для пар колонок без определённой корреляции
  (одна колонка константна — pandas возвращает NaN в r);
- метрики baseline-моделей при единственном классе в фолде CV (precision = NaN);
- Mutual Information для признаков с нулевой дисперсией.

Без `jsonb_safe` запись в JSONB падает с
`psycopg.errors.InvalidTextRepresentation: Token "NaN" is invalid`.
"""
from __future__ import annotations

import math
from typing import Any


def jsonb_safe(value: Any) -> Any:
    """
    Рекурсивно заменяет NaN/Infinity на None в произвольной dict/list/float-структуре.

    Поддерживаемые типы:
    - float (включая np.float64, который isinstance(x, float) возвращает True);
    - dict — рекурсивно по значениям;
    - list — рекурсивно по элементам;
    - всё остальное проходит без изменений.

    Работает «по месту» концептуально, но возвращает новые контейнеры —
    исходный объект не мутируется.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: jsonb_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonb_safe(v) for v in value]
    return value
