"""
Seed-скрипт для справочника quality_rules.

Заливает 12 правил проверки качества данных, описанных в
.knowledge/methods/quality-checks.md. Идемпотентен: при повторном запуске
существующие записи (по уникальному коду) не дублируются и не перезаписываются.
Если потребуется обновить пороги или тексты — это можно сделать прямым SQL
или администраторским эндпоинтом, чтобы не терять историю изменений.

Запуск:
    docker compose exec backend python -m seeds.seed_quality_rules
или через Makefile:
    make seed-rules
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.db import SessionLocal
from app.models.quality_rule import QualityRule

# Полный набор правил. Поле `code` уникально и используется в quality_checker
# как идентификатор правила. Severity берётся из quality-checks.md как
# источника истины.
RULES: list[dict] = [
    # ── Critical ────────────────────────────────────────────────────────────
    {
        "code": "TARGET_MISSING",
        "name": "Пропуски в целевом столбце",
        "description": (
            "Срабатывает, если доля пропусков в столбце target превышает порог. "
            "Наблюдения с пропуском в target нельзя использовать для обучения "
            "с учителем — их придётся отбросить или восстановить."
        ),
        "thresholds": {"max_target_missing_pct": 0.05},
        "severity": "critical",
    },
    {
        "code": "LEAKAGE_SUSPICION",
        "name": "Подозрение на утечку данных",
        "description": (
            "Признак имеет аномально высокую связь с target: |Pearson r| > 0.95 "
            "или mutual information > 0.9. Часто это означает, что признак "
            "вычислен из target или содержит «будущие» данные."
        ),
        "thresholds": {
            "max_correlation_with_target": 0.95,
            "max_mutual_info_with_target": 0.9,
        },
        "severity": "critical",
    },
    # ── Warning ─────────────────────────────────────────────────────────────
    {
        "code": "HIGH_MISSING",
        "name": "Высокая доля пропусков в столбце",
        "description": (
            "Доля пропусков в столбце превышает порог. Перед обучением такие "
            "столбцы требуют обработки (импутация, удаление, индикатор пропуска)."
        ),
        "thresholds": {"max_col_missing_pct": 0.3},
        "severity": "warning",
    },
    {
        "code": "DUPLICATES",
        "name": "Дубликаты строк",
        "description": (
            "Доля полностью повторяющихся строк превышает порог. Дубликаты "
            "искажают оценки качества модели и могут попасть и в train, и в test."
        ),
        "thresholds": {"max_duplicates_pct": 0.05},
        "severity": "warning",
    },
    {
        "code": "IMBALANCE_BINARY",
        "name": "Дисбаланс классов в бинарной классификации",
        "description": (
            "Соотношение размеров классов хуже заданного порога. Требует "
            "стратегий балансировки (oversampling, class_weight) и метрик, "
            "устойчивых к дисбалансу (F1, ROC-AUC)."
        ),
        "thresholds": {"max_imbalance_ratio": 10.0},
        "severity": "warning",
    },
    {
        "code": "IMBALANCE_MULTICLASS",
        "name": "Недостаточно объектов в классе",
        "description": (
            "В наименьшем классе слишком мало наблюдений для надёжного обучения. "
            "Рекомендуется собрать больше данных или объединить редкие классы."
        ),
        "thresholds": {"min_class_size": 50},
        "severity": "warning",
    },
    {
        "code": "SMALL_DATASET",
        "name": "Слишком маленький датасет",
        "description": (
            "Число строк ниже порога. На малых выборках кросс-валидация "
            "ненадёжна и любые метрики имеют большую дисперсию."
        ),
        "thresholds": {"min_rows": 100},
        "severity": "warning",
    },
    {
        "code": "TOO_FEW_FEATURES",
        "name": "Недостаточно признаков",
        "description": (
            "Число признаков (без target) ниже порога. С таким набором сложно "
            "построить качественную модель — стоит подумать о feature engineering."
        ),
        "thresholds": {"min_cols": 3},
        "severity": "warning",
    },
    # ── Info ────────────────────────────────────────────────────────────────
    {
        "code": "LOW_VARIANCE",
        "name": "Признак с низкой вариативностью",
        "description": (
            "Числовой признак почти константен (std/|mean| ниже порога) или "
            "категориальный имеет нормированную энтропию ниже порога. "
            "Малоинформативен для модели."
        ),
        "thresholds": {"min_cv": 0.01, "min_normalized_entropy": 0.1},
        "severity": "info",
    },
    {
        "code": "HIGH_CARDINALITY",
        "name": "Высокая кардинальность",
        "description": (
            "Доля уникальных значений в категориальном признаке превышает порог. "
            "Возможно, это идентификатор — рекомендуется исключить из обучения."
        ),
        "thresholds": {"max_cardinality_ratio": 0.5},
        "severity": "info",
    },
    {
        "code": "OUTLIERS",
        "name": "Выбросы в численном признаке",
        "description": (
            "Доля выбросов превышает порог. Метод определения выбирается "
            "автоматически: Z-score для нормальных распределений, "
            "иначе IQR-метод Тьюки (1977)."
        ),
        "thresholds": {"max_outliers_pct": 0.05},
        "severity": "info",
    },
    {
        "code": "DATE_NOT_PARSED",
        "name": "Дата хранится как строка",
        "description": (
            "Строковый признак выглядит как дата (доля успешного парсинга через "
            "pandas.to_datetime превышает порог), но не приведён к типу datetime. "
            "Рекомендуется преобразование для извлечения признаков (год, месяц и т.д.)."
        ),
        "thresholds": {"min_date_parse_rate": 0.9},
        "severity": "info",
    },
]


def seed() -> int:
    """
    Идемпотентно записывает справочник правил.

    Использует PostgreSQL-специфичный INSERT ... ON CONFLICT (code) DO NOTHING,
    чтобы повторный запуск не падал на UNIQUE-нарушении и не перезаписывал
    существующие записи (если администратор уже изменил пороги через UI/SQL).

    Returns:
        Сколько правил действительно вставлено в этом запуске.
    """
    with SessionLocal() as db:
        existing_codes = set(db.execute(select(QualityRule.code)).scalars().all())
        new_rules = [r for r in RULES if r["code"] not in existing_codes]
        if new_rules:
            db.execute(
                insert(QualityRule)
                .values(new_rules)
                .on_conflict_do_nothing(index_elements=["code"])
            )
            db.commit()
        return len(new_rules)


def main() -> None:
    inserted = seed()
    total = len(RULES)
    print(f"seed_quality_rules: inserted {inserted} new of {total} rules")


if __name__ == "__main__":
    main()
