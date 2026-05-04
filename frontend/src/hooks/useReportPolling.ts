// Хук отслеживания статуса PDF-отчёта. Дёргает GET /reports/{id} каждые
// 2 секунды, пока status ∈ {pending, running}; останавливается на
// success/failed. В отличие от useBaselinePolling 404 здесь не нормальное
// состояние «not started», а ошибка (id, который мы держим в state, должен
// существовать) — пробрасываем как есть.
import { useQuery } from "@tanstack/react-query";
import { reportsApi } from "../api/reports";
import type { Report } from "../types/report";

const POLL_INTERVAL_MS = 2000;

export function useReportPolling(reportId: string | null) {
  return useQuery<Report>({
    queryKey: ["report", reportId],
    queryFn: () => reportsApi.get(reportId!),
    enabled: !!reportId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "pending" || status === "running"
        ? POLL_INTERVAL_MS
        : false;
    },
    refetchIntervalInBackground: false,
  });
}
