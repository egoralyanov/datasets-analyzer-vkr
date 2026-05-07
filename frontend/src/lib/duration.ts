// Длительность между двумя ISO-метками для UI /history.
//
// Sub-second интервалы (профайлинг маленьких датасетов с warm-cache)
// показываем как «<1 сек», иначе округляем до ближайшей секунды.
// До Sprint 6 Phase 8.1 функция округляла раньше времени, и любой
// интервал < 500 мс выводился как «0 сек» — выглядело как баг.
//
// Возвращает null, если finished_at не задан или метки невалидны —
// вызывающий код в History.tsx скрывает duration-сегмент в этом случае.
export function computeDuration(
  startedIso: string,
  finishedIso: string | null,
): string | null {
  if (!finishedIso) return null;
  const startMs = Date.parse(startedIso);
  const endMs = Date.parse(finishedIso);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null;
  const diffMs = Math.max(0, endMs - startMs);
  if (diffMs < 1000) return "<1 сек";
  const seconds = Math.round(diffMs / 1000);
  if (seconds < 60) return `${seconds} сек`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest === 0 ? `${minutes} мин` : `${minutes} мин ${rest} сек`;
}
