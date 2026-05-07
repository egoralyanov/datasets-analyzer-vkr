// Единая точка управления генерацией и скачиванием PDF-отчёта. Хранит
// reportId, делает polling статуса и запускает start/download mutations.
//
// Sprint 6, Phase 2: lifted-state из ReportDownloadCard, чтобы sticky-bar
// мог триггерить генерацию и карточка одновременно отображала статус.
//
// Известное ограничение из Sprint 4 сохраняется: при перезагрузке страницы
// reportId сбрасывается в null даже если success-отчёт уже есть в БД (нет
// latest-report endpoint'а).
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import { reportsApi } from "../api/reports";
import { useReportPolling } from "./useReportPolling";
import type {
  ReportConflictResponse,
  ReportStatus,
} from "../types/report";

export interface ReportActions {
  reportId: string | null;
  status: ReportStatus | null;
  fileSizeBytes: number | null;
  pollingError: string | null;
  flowError: string | null;
  isStarting: boolean;
  isDownloading: boolean;
  start: () => void;
  download: () => void;
  retry: () => void;
}

const ERROR_AUTO_DISMISS_MS = 5000;

export function useReportActions(
  analysisId: string | undefined,
): ReportActions {
  const [reportId, setReportId] = useState<string | null>(null);
  const [flowError, setFlowError] = useState<string | null>(null);
  const polling = useReportPolling(reportId);

  const startMutation = useMutation({
    mutationFn: () => reportsApi.create(analysisId!),
    onMutate: () => setFlowError(null),
    onSuccess: (resp) => setReportId(resp.id),
    onError: (err: unknown) => {
      const axiosErr = err as AxiosError<ReportConflictResponse>;
      const conflict = axiosErr.response?.data;
      // 409 + report_in_progress: подцепляемся к polling существующего отчёта.
      if (
        axiosErr.response?.status === 409 &&
        conflict?.reason === "report_in_progress" &&
        conflict.report_id
      ) {
        setReportId(conflict.report_id);
        return;
      }
      const message =
        conflict?.detail ??
        (err instanceof Error
          ? err.message
          : "Не удалось запустить генерацию отчёта");
      setFlowError(message);
      window.setTimeout(() => setFlowError(null), ERROR_AUTO_DISMISS_MS);
    },
  });

  const downloadMutation = useMutation({
    mutationFn: () => reportsApi.download(reportId!),
    onSuccess: ({ blob, filename }) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error ? err.message : "Не удалось скачать отчёт";
      setFlowError(message);
      window.setTimeout(() => setFlowError(null), ERROR_AUTO_DISMISS_MS);
    },
  });

  const start = () => {
    if (analysisId) startMutation.mutate();
  };

  return {
    reportId,
    status: polling.data?.status ?? null,
    fileSizeBytes: polling.data?.file_size_bytes ?? null,
    pollingError: polling.data?.error ?? null,
    flowError,
    isStarting: startMutation.isPending,
    isDownloading: downloadMutation.isPending,
    start,
    download: () => {
      if (reportId) downloadMutation.mutate();
    },
    // retry — после failed: новый отчёт = новый reportId, старый failed остаётся.
    retry: () => {
      setReportId(null);
      start();
    },
  };
}
