// Единый источник человекочитаемых названий типов ML-задач.
//
// До Sprint 6 Phase 5.2 словарь был задублирован в TaskRecommendationCard.tsx
// и Landing.tsx; теперь оба читают его отсюда. Здесь же — helper, который
// безопасно резолвит код в подпись (null/undefined/неизвестный код → null).
import type { TaskTypeCode } from "../types/analysis";

export type { TaskTypeCode } from "../types/analysis";

export const TASK_TYPE_LABEL: Record<TaskTypeCode, string> = {
  BINARY_CLASSIFICATION: "Бинарная классификация",
  MULTICLASS_CLASSIFICATION: "Многоклассовая классификация",
  REGRESSION: "Регрессия",
  CLUSTERING: "Кластеризация",
  NOT_READY: "Данные не готовы для ML",
};

// Возвращает подпись или null, если код не задан/неизвестен. Удобно для
// условного рендера в UI: `label && <span>{label}</span>`.
export function getTaskTypeLabel(
  code: TaskTypeCode | string | null | undefined,
): string | null {
  if (!code) return null;
  return TASK_TYPE_LABEL[code as TaskTypeCode] ?? null;
}
