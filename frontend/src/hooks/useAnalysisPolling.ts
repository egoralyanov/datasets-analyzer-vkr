// Хук для отслеживания статуса анализа: дёргает GET /analyses/{id}
// каждые 2 секунды, пока статус pending/running. Останавливается на done/failed.
import { useQuery } from "@tanstack/react-query";
import { analysesApi } from "../api/analyses";
import type { Analysis } from "../types/analysis";

const POLL_INTERVAL_MS = 2000;

export function useAnalysisPolling(analysisId: string | undefined) {
  return useQuery<Analysis>({
    queryKey: ["analysis", analysisId],
    queryFn: () => analysesApi.get(analysisId!),
    enabled: !!analysisId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return POLL_INTERVAL_MS;
      // Пока бэк работает — спрашиваем каждые 2 секунды.
      return data.status === "pending" || data.status === "running"
        ? POLL_INTERVAL_MS
        : false;
    },
    refetchIntervalInBackground: false,
  });
}
