"""
Сервис проверки качества данных: применение 12 правил к meta-features и df.

Полное описание правил, формул и порогов — в .knowledge/methods/quality-checks.md
(этот файл войдёт в Главу 2 РПЗ как раздел «Контроль качества данных»).

Принципы реализации:
- Каждое правило — отдельная функция-чекер (Правило 6 из CLAUDE.md), что
  делает каждое правило защитимым на ГЭК и легко тестируемым (Phase 4).
- Чекеры **не пересчитывают** статистики, которые уже есть в meta-features —
  это и согласованнее, и быстрее: профайлер посчитал один раз, дальше только
  применяем пороги.
- Пороги хранятся в таблице quality_rules.thresholds (JSONB) и подгружаются
  в run_quality_checks. Это позволяет администратору менять чувствительность
  правил без редеплоя — требование .knowledge/methods/quality-checks.md.
- Чекер возвращает 0..N черновиков FlagDraft. 0 — если правило не
  сработало; N>1 — если правило срабатывает по разным колонкам (например,
  HIGH_MISSING для каждой проблемной колонки даёт отдельный флаг).
- Главная функция run_quality_checks превращает черновики в ORM-объекты
  QualityFlag, привязывая их к analysis_id и rule_id.

Severity не задаётся в чекере — он берётся из БД (поле quality_rules.severity).
Это нормализация: при изменении severity правила исторические флаги
автоматически переедут на новый уровень.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.quality_flag import QualityFlag
from app.models.quality_rule import QualityRule

# Эвристика DATE_NOT_PARSED: проверяем первые DATE_PARSE_SAMPLE значений колонки.
DATE_PARSE_SAMPLE = 100


@dataclass
class FlagDraft:
    """
    Промежуточный объект — результат работы одного чекера.

    Хранит код правила, текст для пользователя и контекст срабатывания
    (детали для UI «Подробнее»). Превращается в ORM QualityFlag в
    run_quality_checks, где известен analysis_id и rule_id.
    """

    rule_code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


# =============================================================================
#                               1. CRITICAL
# =============================================================================


def check_target_missing(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило TARGET_MISSING: пропуски в целевом столбце.

    Условие: доля пропусков в target превышает порог `max_target_missing_pct`
    (по умолчанию 5%). Наблюдения с NaN в target нельзя использовать для
    обучения с учителем — их придётся отбросить или восстанавливать.

    См. .knowledge/methods/quality-checks.md, правило 2.

    Args:
        df: DataFrame.
        target_col: имя целевого столбца (или None).
        meta: meta-features из profiler.compute_meta_features.
        thresholds: пороги правила из quality_rules.thresholds.

    Returns:
        Список из 0..1 черновиков (правило не работает без target_col).
    """
    if target_col is None or target_col not in df.columns:
        return []
    missing_pct = float(df[target_col].isna().mean())
    threshold = float(thresholds.get("max_target_missing_pct", 0.05))
    if missing_pct <= threshold:
        return []
    return [
        FlagDraft(
            rule_code="TARGET_MISSING",
            message=(
                f"В целевом столбце «{target_col}» {missing_pct:.1%} пропусков "
                f"(порог {threshold:.0%}). Эти наблюдения непригодны для обучения с учителем."
            ),
            context={
                "column": target_col,
                "missing_pct": missing_pct,
                "threshold": threshold,
            },
        )
    ]


def check_leakage_suspicion(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило LEAKAGE_SUSPICION: подозрительно высокая связь признака с target.

    Условие: для какого-либо признака |Pearson r| превышает
    `max_correlation_with_target` (по умолчанию 0.95) ИЛИ Mutual Information
    превышает `max_mutual_info_with_target` (по умолчанию 0.9). Это часто
    означает, что признак вычислен из target или содержит «будущие» данные.

    На каждый подозрительный признак создаётся отдельный флаг — пользователь
    должен видеть конкретный список колонок, который требует проверки.

    См. .knowledge/methods/quality-checks.md, правило 9.

    Args:
        df: DataFrame.
        target_col: имя целевого столбца.
        meta: meta-features (содержит target_correlation_by_column и
            target_mutual_information_by_column).
        thresholds: пороги правила.

    Returns:
        По одному флагу на каждый подозрительный признак.
    """
    if target_col is None:
        return []
    corr_threshold = float(thresholds.get("max_correlation_with_target", 0.95))
    mi_threshold = float(thresholds.get("max_mutual_info_with_target", 0.9))

    drafts: list[FlagDraft] = []

    # Корреляция Пирсона имеет смысл только для пар числовых.
    corr_by_col = meta.get("target_correlation_by_column") or {}
    for col, corr in corr_by_col.items():
        if abs(float(corr)) > corr_threshold:
            drafts.append(
                FlagDraft(
                    rule_code="LEAKAGE_SUSPICION",
                    message=(
                        f"Признак «{col}» имеет корреляцию Пирсона {float(corr):.3f} с target — "
                        "возможна утечка данных."
                    ),
                    context={
                        "column": col,
                        "method": "pearson",
                        "correlation": float(corr),
                        "threshold": corr_threshold,
                    },
                )
            )

    # MI работает с любыми типами признаков (после LabelEncoder).
    mi_by_col = meta.get("target_mutual_information_by_column") or {}
    flagged_cols = {d.context["column"] for d in drafts}
    for col, mi in mi_by_col.items():
        if float(mi) > mi_threshold and col not in flagged_cols:
            drafts.append(
                FlagDraft(
                    rule_code="LEAKAGE_SUSPICION",
                    message=(
                        f"Признак «{col}» имеет mutual information {float(mi):.3f} с target — "
                        "возможна утечка данных."
                    ),
                    context={
                        "column": col,
                        "method": "mutual_information",
                        "mutual_info": float(mi),
                        "threshold": mi_threshold,
                    },
                )
            )
    return drafts


# =============================================================================
#                               2. WARNING
# =============================================================================


def check_high_missing(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило HIGH_MISSING: высокая доля пропусков в столбце.

    Условие: доля пропусков в каком-либо признаке превышает порог
    `max_col_missing_pct` (по умолчанию 30%). target проверяется отдельным
    правилом TARGET_MISSING, поэтому здесь исключается.

    На каждую проблемную колонку — отдельный флаг.

    См. .knowledge/methods/quality-checks.md, правило 1.
    """
    threshold = float(thresholds.get("max_col_missing_pct", 0.3))
    missing_by_col = meta.get("missing_by_column", {})
    drafts: list[FlagDraft] = []
    for col, pct in missing_by_col.items():
        if col == target_col:
            continue
        pct_f = float(pct)
        if pct_f > threshold:
            drafts.append(
                FlagDraft(
                    rule_code="HIGH_MISSING",
                    message=(
                        f"В столбце «{col}» {pct_f:.1%} пропусков (порог {threshold:.0%}) — "
                        "перед обучением требуется обработка."
                    ),
                    context={
                        "column": col,
                        "missing_pct": pct_f,
                        "threshold": threshold,
                    },
                )
            )
    return drafts


def check_duplicates(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило DUPLICATES: доля полных дубликатов строк превышает порог.

    Условие: duplicate_rows_pct > `max_duplicates_pct` (по умолчанию 5%).
    Дубликаты искажают оценки качества модели и могут попасть и в train, и в test.

    См. .knowledge/methods/quality-checks.md, правило 3.
    """
    threshold = float(thresholds.get("max_duplicates_pct", 0.05))
    pct = float(meta.get("duplicate_rows_pct", 0.0))
    if pct <= threshold:
        return []
    n_dup = int(round(pct * meta.get("n_rows", 0)))
    return [
        FlagDraft(
            rule_code="DUPLICATES",
            message=(
                f"Доля полных дубликатов строк составляет {pct:.1%} "
                f"(порог {threshold:.0%}, {n_dup} строк)."
            ),
            context={
                "duplicate_pct": pct,
                "threshold": threshold,
                "n_duplicates": n_dup,
            },
        )
    ]


def check_imbalance_binary(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило IMBALANCE_BINARY: дисбаланс в бинарной классификации.

    Условие: target — категориальный с ровно двумя классами, а соотношение
    max/min размера класса превышает `max_imbalance_ratio` (по умолчанию 10:1).

    См. .knowledge/methods/quality-checks.md, правило 6.
    """
    if meta.get("target_kind") != "categorical":
        return []
    counts = meta.get("target_value_counts") or {}
    if len(counts) != 2:
        return []
    threshold = float(thresholds.get("max_imbalance_ratio", 10.0))
    ratio = float(meta.get("target_imbalance_ratio") or 0.0)
    if ratio <= threshold:
        return []
    return [
        FlagDraft(
            rule_code="IMBALANCE_BINARY",
            message=(
                f"Сильный дисбаланс классов в target: {ratio:.1f}:1 "
                f"(порог {threshold:.0f}:1). "
                "Используйте F1/ROC-AUC и стратегии балансировки (oversampling, class_weight)."
            ),
            context={
                "column": target_col,
                "ratio": ratio,
                "threshold": threshold,
                "value_counts": counts,
            },
        )
    ]


def check_imbalance_multiclass(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило IMBALANCE_MULTICLASS: недостаточно объектов в классе.

    Условие: target — категориальный с ≥ 3 классами, размер наименьшего
    класса меньше `min_class_size` (по умолчанию 50). Это правило именно
    про абсолютный размер: даже при сбалансированных пропорциях, если
    каждого класса < 50, надёжно учиться нельзя.

    См. .knowledge/methods/quality-checks.md, правило 7.
    """
    if meta.get("target_kind") != "categorical":
        return []
    counts = meta.get("target_value_counts") or {}
    if len(counts) < 3:
        return []
    threshold = int(thresholds.get("min_class_size", 50))
    min_label, min_size = min(counts.items(), key=lambda kv: kv[1])
    if min_size >= threshold:
        return []
    return [
        FlagDraft(
            rule_code="IMBALANCE_MULTICLASS",
            message=(
                f"В классе «{min_label}» всего {int(min_size)} объектов "
                f"(порог {threshold}) — недостаточно для надёжного обучения."
            ),
            context={
                "column": target_col,
                "min_class": str(min_label),
                "min_class_size": int(min_size),
                "threshold": threshold,
                "value_counts": counts,
            },
        )
    ]


def check_small_dataset(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило SMALL_DATASET: слишком маленький датасет.

    Условие: число строк (оригинального датасета, до сэмплирования) ниже
    порога `min_rows` (по умолчанию 100).

    См. .knowledge/methods/quality-checks.md, правило 10.
    """
    threshold = int(thresholds.get("min_rows", 100))
    sampling = meta.get("sampling") or {}
    n_rows = int(sampling.get("original_size") or meta.get("n_rows", 0))
    if n_rows >= threshold:
        return []
    return [
        FlagDraft(
            rule_code="SMALL_DATASET",
            message=(
                f"В датасете {n_rows} строк (порог {threshold}) — на малых выборках "
                "оценки качества имеют большую дисперсию."
            ),
            context={"n_rows": n_rows, "threshold": threshold},
        )
    ]


def check_too_few_features(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило TOO_FEW_FEATURES: недостаточно признаков (без target).

    Условие: число столбцов после исключения target меньше `min_cols`
    (по умолчанию 3).

    См. .knowledge/methods/quality-checks.md, правило 11.
    """
    threshold = int(thresholds.get("min_cols", 3))
    n_cols = int(meta.get("n_cols", 0))
    n_features = n_cols - (1 if target_col and target_col in df.columns else 0)
    if n_features >= threshold:
        return []
    return [
        FlagDraft(
            rule_code="TOO_FEW_FEATURES",
            message=(
                f"Признаков всего {n_features} (порог {threshold}) — "
                "качественную модель построить будет сложно."
            ),
            context={"n_features": n_features, "threshold": threshold},
        )
    ]


# =============================================================================
#                                 3. INFO
# =============================================================================


def check_low_variance(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило LOW_VARIANCE: признаки с низкой вариативностью.

    Условие:
    - для числовых: std/|mean| < `min_cv` (по умолчанию 0.01) — почти константа;
    - для категориальных: нормированная энтропия < `min_normalized_entropy`
      (по умолчанию 0.1) — одно значение доминирует.

    На каждую малоинформативную колонку — отдельный флаг.

    См. .knowledge/methods/quality-checks.md, правило 4.
    """
    drafts: list[FlagDraft] = []
    numeric_cv_threshold = float(thresholds.get("min_cv", 0.01))
    entropy_threshold = float(thresholds.get("min_normalized_entropy", 0.1))

    for col in meta.get("low_variance_numeric_cols", []) or []:
        drafts.append(
            FlagDraft(
                rule_code="LOW_VARIANCE",
                message=(
                    f"Числовой признак «{col}» почти константен "
                    "(коэффициент вариации ниже порога) — малоинформативен."
                ),
                context={
                    "column": col,
                    "type": "numeric",
                    "threshold_cv": numeric_cv_threshold,
                },
            )
        )
    for col in meta.get("low_variance_categorical_cols", []) or []:
        entropy = (meta.get("entropy_by_column") or {}).get(col)
        drafts.append(
            FlagDraft(
                rule_code="LOW_VARIANCE",
                message=(
                    f"Категориальный признак «{col}» имеет нормированную энтропию "
                    f"{(entropy or 0.0):.3f} — одно значение доминирует."
                ),
                context={
                    "column": col,
                    "type": "categorical",
                    "normalized_entropy": float(entropy) if entropy is not None else None,
                    "threshold_entropy": entropy_threshold,
                },
            )
        )
    return drafts


def check_high_cardinality(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило HIGH_CARDINALITY: высокая кардинальность категориального признака.

    Условие: доля уникальных значений выше `max_cardinality_ratio`
    (по умолчанию 0.5). Возможно, это идентификатор — обычно нужно
    исключить из обучения.

    См. .knowledge/methods/quality-checks.md, правило 5.
    """
    threshold = float(thresholds.get("max_cardinality_ratio", 0.5))
    drafts: list[FlagDraft] = []
    cardinality_by_col = meta.get("cardinality_by_column") or {}
    for col in meta.get("high_cardinality_cols", []) or []:
        ratio = float(cardinality_by_col.get(col, 0.0))
        drafts.append(
            FlagDraft(
                rule_code="HIGH_CARDINALITY",
                message=(
                    f"Признак «{col}»: {ratio:.1%} уникальных значений "
                    f"(порог {threshold:.0%}) — возможно, идентификатор."
                ),
                context={
                    "column": col,
                    "cardinality_ratio": ratio,
                    "threshold": threshold,
                },
            )
        )
    return drafts


def check_outliers(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило OUTLIERS: выбросов в числовом признаке больше порога.

    Условие: доля выбросов превышает `max_outliers_pct` (по умолчанию 5%).
    Метод обнаружения выбран автоматически профайлером (Z-score для
    нормальных распределений, IQR-метод Тьюки иначе) — здесь только
    применяем порог к посчитанной доле.

    См. .knowledge/methods/quality-checks.md, правило 8.
    """
    threshold = float(thresholds.get("max_outliers_pct", 0.05))
    drafts: list[FlagDraft] = []
    outliers_by_col = meta.get("outliers_by_column") or {}
    for col, pct in outliers_by_col.items():
        if col == target_col:
            continue
        pct_f = float(pct)
        if pct_f > threshold:
            drafts.append(
                FlagDraft(
                    rule_code="OUTLIERS",
                    message=(
                        f"В признаке «{col}» {pct_f:.1%} выбросов (порог {threshold:.0%}) — "
                        "рассмотрите фильтрацию или преобразование."
                    ),
                    context={
                        "column": col,
                        "outliers_pct": pct_f,
                        "threshold": threshold,
                    },
                )
            )
    return drafts


def check_date_not_parsed(
    df: pd.DataFrame,
    target_col: str | None,
    meta: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[FlagDraft]:
    """
    Правило DATE_NOT_PARSED: строковая колонка похожа на дату.

    Эвристика: для каждой object-колонки берём первые DATE_PARSE_SAMPLE
    непустых значений и пробуем `pd.to_datetime(errors='coerce')`. Если
    доля успешного парсинга превышает `min_date_parse_rate` (по умолчанию
    90%) — флаг. Преобразование в datetime раскрывает дополнительные
    признаки (год, месяц, день недели).

    См. .knowledge/methods/quality-checks.md, правило 12.
    """
    threshold = float(thresholds.get("min_date_parse_rate", 0.9))
    drafts: list[FlagDraft] = []
    for col in df.columns:
        # Проверяем только object/string-колонки (datetime уже распарсены).
        if df[col].dtype.kind not in {"O", "U", "S"}:
            continue
        sample = df[col].dropna().astype(str).head(DATE_PARSE_SAMPLE)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        parse_rate = float(parsed.notna().mean())
        if parse_rate >= threshold:
            drafts.append(
                FlagDraft(
                    rule_code="DATE_NOT_PARSED",
                    message=(
                        f"Признак «{col}» выглядит как дата ({parse_rate:.0%} значений "
                        "успешно парсится), но хранится как строка."
                    ),
                    context={
                        "column": col,
                        "parse_rate": parse_rate,
                        "threshold": threshold,
                        "sample_size": int(sample.size),
                    },
                )
            )
    return drafts


# =============================================================================
#                              4. ОРКЕСТРАТОР
# =============================================================================


# Реестр чекеров: code → функция. Используется в run_quality_checks.
CHECKERS: dict[str, Callable[..., list[FlagDraft]]] = {
    "TARGET_MISSING": check_target_missing,
    "LEAKAGE_SUSPICION": check_leakage_suspicion,
    "HIGH_MISSING": check_high_missing,
    "DUPLICATES": check_duplicates,
    "IMBALANCE_BINARY": check_imbalance_binary,
    "IMBALANCE_MULTICLASS": check_imbalance_multiclass,
    "SMALL_DATASET": check_small_dataset,
    "TOO_FEW_FEATURES": check_too_few_features,
    "LOW_VARIANCE": check_low_variance,
    "HIGH_CARDINALITY": check_high_cardinality,
    "OUTLIERS": check_outliers,
    "DATE_NOT_PARSED": check_date_not_parsed,
}


def run_quality_checks(
    df: pd.DataFrame,
    target_col: str | None,
    meta_features: dict[str, Any],
    analysis_id,
    db: Session,
) -> list[QualityFlag]:
    """
    Применяет все активные правила и возвращает список ORM-объектов QualityFlag.

    Алгоритм:
    1. Подгружаем активные правила из БД (с порогами и rule_id).
    2. Для каждого правила, у которого есть зарегистрированный чекер,
       применяем функцию-проверку и собираем черновики FlagDraft.
    3. Превращаем каждый черновик в ORM QualityFlag, привязывая к analysis_id
       и rule_id (severity не дублируется в флаге — берётся через JOIN).

    Возвращённые объекты ещё не сохранены — их сохраняет
    quality_flag_repo.bulk_create_flags в API/orchestrator-слое.

    Args:
        df: DataFrame, на котором запускался профайлер.
        target_col: имя целевого столбца или None.
        meta_features: результат profiler.compute_meta_features.
        analysis_id: UUID анализа, к которому будут привязаны флаги.
        db: SQLAlchemy-сессия — для подгрузки активных правил.

    Returns:
        Список QualityFlag, готовых к bulk_insert.
    """
    rules = (
        db.execute(select(QualityRule).where(QualityRule.is_active.is_(True)))
        .scalars()
        .all()
    )
    rules_by_code = {r.code: r for r in rules}

    flags: list[QualityFlag] = []
    for code, checker in CHECKERS.items():
        rule = rules_by_code.get(code)
        if rule is None:
            # Правило отсутствует или деактивировано — пропускаем чекер.
            continue
        for draft in checker(df, target_col, meta_features, rule.thresholds or {}):
            flags.append(
                QualityFlag(
                    analysis_id=analysis_id,
                    rule_id=rule.id,
                    message=draft.message,
                    context=draft.context,
                )
            )
    return flags
