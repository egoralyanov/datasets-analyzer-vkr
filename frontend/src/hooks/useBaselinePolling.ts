// Хук для отслеживания состояния baseline-обучения. Дёргает
// GET /analyses/{id}/baseline каждые 2 секунды, пока baseline_status='running'.
// Останавливается на done/failed/404. См. useAnalysisPolling — паттерн идентичный.
import { useQuery } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import { analysesApi } from "../api/analyses";
import type { BaselineResponse } from "../types/analysis";

const POLL_INTERVAL_MS = 2000;

// Backend возвращает 404 для baseline_status='not_started' — это нормальное
// состояние «ещё не запускали», а не ошибка. Превращаем такой ответ в
// «синтетический» BaselineResponse, чтобы остальной код работал единообразно.
const NOT_STARTED_RESPONSE: BaselineResponse = {
  baseline_status: "not_started",
  baseline: null,
  baseline_error: null,
};

export function useBaselinePolling(analysisId: string | undefined) {
  return useQuery<BaselineResponse>({
    queryKey: ["baseline", analysisId],
    queryFn: async () => {
      try {
        return await analysesApi.getBaseline(analysisId!);
      } catch (err) {
        const axiosErr = err as AxiosError;
        if (axiosErr.response?.status === 404) {
          return NOT_STARTED_RESPONSE;
        }
        throw err;
      }
    },
    enabled: !!analysisId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return false;
      return data.baseline_status === "running" ? POLL_INTERVAL_MS : false;
    },
    refetchIntervalInBackground: false,
  });
}
