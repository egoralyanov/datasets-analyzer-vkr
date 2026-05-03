"""
Схемы рекомендации типа ML-задачи (Слой 1 + Слой 2 гибрид).

Содержит две сущности:

1. `TaskRecommendation` — Pydantic-модель, итог `recommend_task()`. Сериализуется
   в JSONB-поле `analysis_results.task_recommendation` через `model_dump()`
   (см. Phase 6 интеграции в analysis_service). Поэтому все поля должны быть
   JSON-совместимыми (никаких UUID, datetime — только str/int/float/bool/dict/list).

2. `RulesResult` — внутренний dataclass, который возвращает `apply_rules()`.
   Не уезжает за пределы task_recommender (не сериализуется в БД, не уходит в API).
   Используется только для передачи результата Слоя 1 в логику гибрида.

Возможные значения `task_type_code` (5 финальных + 6-я метка только в applied_rules):
- BINARY_CLASSIFICATION
- MULTICLASS_CLASSIFICATION
- REGRESSION
- CLUSTERING
- NOT_READY
- DIMENSIONALITY_REDUCTION — НЕ как самостоятельный task_type_code, а как метка
  правила NO_TARGET_HIGH_DIM в applied_rules (вспомогательный шаг перед CLUSTERING).

См. `.knowledge/methods/recommender-rules.md` (дерево правил) и
`.knowledge/methods/recommender-ml.md` (логика гибрида).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Коды task_type, которые могут быть финальными в TaskRecommendation.
# DIMENSIONALITY_REDUCTION в этот список не входит (см. модульный docstring).
TaskTypeCode = Literal[
    "BINARY_CLASSIFICATION",
    "MULTICLASS_CLASSIFICATION",
    "REGRESSION",
    "CLUSTERING",
    "NOT_READY",
]


class AppliedRule(BaseModel):
    """
    Одно сработавшее правило Слоя 1.

    Используется в `applied_rules: list[AppliedRule]` финальной рекомендации.
    `code` — машинный идентификатор (NUMERIC_BINARY_TARGET, BINARY_BALANCED, ...);
    `description` — русский текст для UI.
    """

    code: str
    description: str


class TaskRecommendation(BaseModel):
    """
    Финальная рекомендация типа ML-задачи (источник истины — Слой 1, Слой 2 или гибрид).

    Сериализуется в `analysis_results.task_recommendation` JSONB через `model_dump()`.
    """

    model_config = ConfigDict(extra="forbid")

    task_type_code: TaskTypeCode = Field(
        description="Финальная рекомендация типа задачи. NB: DIMENSIONALITY_REDUCTION "
        "сюда не попадает — он живёт только как код правила в applied_rules."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Уверенность в рекомендации в [0, 1]. Источник зависит от source.",
    )
    source: Literal["rules", "ml", "hybrid"] = Field(
        description="Какой слой принял окончательное решение. "
        "rules — Слой 1; ml — Слой 2; hybrid — Слой 2 поверх непустого результата Слоя 1.",
    )
    applied_rules: list[AppliedRule] = Field(
        default_factory=list,
        description="Все сработавшие правила Слоя 1, включая критические флаги "
        "качества данных (TARGET_MISSING, LEAKAGE_SUSPICION, SMALL_DATASET).",
    )
    ml_probabilities: dict[str, float] | None = Field(
        default=None,
        description="Вероятности классов от Слоя 2. None если source='rules' "
        "или модель недоступна.",
    )
    explanation: str = Field(
        description="Человеческий текст на русском для UI «Почему такая рекомендация»."
    )


@dataclass
class RulesResult:
    """
    Внутренний результат `apply_rules()`. Не сериализуется в БД и не уходит в API —
    превращается в `TaskRecommendation` уже после склейки с результатом Слоя 2.

    Поле `requires_ml=True` означает, что правило явно делегировало решение в Слой 2
    (AMBIGUOUS_NUMERIC_TARGET, HIGH_CARDINALITY_MULTICLASS). Это сильнее, чем порог
    `confidence < 0.7` — даже при высокой confidence Слоя 1 модель должна уточнить.
    """

    task_type_code: str
    confidence: float
    requires_ml: bool
    applied_rules: list[AppliedRule] = field(default_factory=list)
