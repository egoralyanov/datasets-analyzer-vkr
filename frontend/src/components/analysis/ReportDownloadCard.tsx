// Карточка генерации и скачивания PDF-отчёта по анализу. Размещается
// последней в цепочке AnalysisResult (после BaselineCard) — семантически
// PDF-отчёт это финальный шаг работы с анализом.
//
// Состояния (local):
//   idle       — reportId === null, никто ещё не нажимал «Сгенерировать».
//   generating — reportId !== null, статус ∈ {pending, running}.
//   ready      — статус === success.
//   error      — статус === failed.
//
// Известное ограничение: при перезагрузке страницы компонент стартует в idle
// даже если у анализа уже есть готовый success-отчёт в БД. Бэк не отдаёт
// «последний report для analysis_id» (нет такого эндпоинта). 90% сценариев
// закрываются через 409 + reason="report_in_progress" — берём report_id
// из тела ответа и подцепляемся к polling. Edge-case «есть готовый PDF за
// прошлую сессию» решается перегенерацией. Закрытие — направление развития.
//
// Frontend-тесты пока не настроены (Vitest/Testing Library не подключены),
// поэтому для этого компонента покрытия нет. TODO: добавить тест на смену
// состояний при настройке тестовой инфраструктуры.
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import { Download, FileText, Loader2, Play, RefreshCw, XCircle } from "lucide-react";
import { reportsApi } from "../../api/reports";
import { useReportPolling } from "../../hooks/useReportPolling";
import type { ReportConflictResponse, ReportStatus } from "../../types/report";
import type { AnalysisStatus } from "../../types/analysis";

type Props = {
  analysisId: string;
  analysisStatus: AnalysisStatus;
};

const ERROR_AUTO_DISMISS_MS = 5000;

export function ReportDownloadCard({ analysisId, analysisStatus }: Props) {
  const [reportId, setReportId] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const polling = useReportPolling(reportId);
  const status: ReportStatus | null = polling.data?.status ?? null;

  const startMutation = useMutation({
    mutationFn: () => reportsApi.create(analysisId),
    onMutate: () => setLocalError(null),
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
      // Прочие ошибки — кратковременно показываем; кнопка disabled должна
      // была это предотвратить, но если как-то прошло — даём пользователю
      // обратную связь, не висим в немом отказе.
      const message =
        conflict?.detail ??
        (err instanceof Error
          ? err.message
          : "Не удалось запустить генерацию отчёта");
      setLocalError(message);
      window.setTimeout(() => setLocalError(null), ERROR_AUTO_DISMISS_MS);
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
      setLocalError(message);
      window.setTimeout(() => setLocalError(null), ERROR_AUTO_DISMISS_MS);
    },
  });

  const startDisabled = analysisStatus !== "done" || startMutation.isPending;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <FileText className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">PDF-отчёт</h2>
      </div>

      {localError && (
        <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {localError}
        </div>
      )}

      {reportId === null && (
        <IdleView
          onStart={() => startMutation.mutate()}
          disabled={startDisabled}
          isStarting={startMutation.isPending}
          analysisStatus={analysisStatus}
        />
      )}

      {reportId !== null && (status === "pending" || status === "running") && (
        <GeneratingView />
      )}

      {reportId !== null && status === "success" && (
        <ReadyView
          onDownload={() => downloadMutation.mutate()}
          isDownloading={downloadMutation.isPending}
          fileSizeBytes={polling.data?.file_size_bytes ?? null}
        />
      )}

      {reportId !== null && status === "failed" && (
        <FailedView
          errorMessage={polling.data?.error ?? null}
          onRetry={() => {
            // Новый Report — старый failed остаётся в БД (тот же паттерн,
            // что в BaselineCard для retry).
            setReportId(null);
            startMutation.mutate();
          }}
          isRetrying={startMutation.isPending}
        />
      )}
    </section>
  );
}

// =============================================================================
//                               СОСТОЯНИЯ
// =============================================================================

function IdleView({
  onStart,
  disabled,
  isStarting,
  analysisStatus,
}: {
  onStart: () => void;
  disabled: boolean;
  isStarting: boolean;
  analysisStatus: AnalysisStatus;
}) {
  return (
    <div className="mt-4 space-y-4">
      <p className="text-sm text-slate-600">
        Сводка датасета, флаги качества, распределения, рекомендация и метрики
        baseline в одном PDF — удобно сохранить или показать руководителю.
        Обычно занимает 15–30 секунд.
      </p>
      <button
        type="button"
        onClick={onStart}
        disabled={disabled}
        className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-400"
      >
        {isStarting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        Сгенерировать отчёт
      </button>
      {analysisStatus !== "done" && (
        <p className="text-xs text-slate-500">
          Кнопка станет активной после завершения анализа.
        </p>
      )}
    </div>
  );
}

function GeneratingView() {
  return (
    <div className="mt-4 flex items-start gap-3 rounded-md border border-blue-200 bg-blue-50 p-4">
      <Loader2 className="mt-0.5 h-5 w-5 animate-spin text-blue-600" />
      <div className="text-sm">
        <p className="font-medium text-blue-900">Генерация отчёта…</p>
        <p className="mt-1 text-blue-800">
          Рендерим графики matplotlib и собираем PDF через WeasyPrint. Обычно
          15–30 секунд.
        </p>
      </div>
    </div>
  );
}

function ReadyView({
  onDownload,
  isDownloading,
  fileSizeBytes,
}: {
  onDownload: () => void;
  isDownloading: boolean;
  fileSizeBytes: number | null;
}) {
  return (
    <div className="mt-4 space-y-3">
      <p className="text-sm text-slate-700">
        Отчёт готов
        {fileSizeBytes ? (
          <span className="ml-1 text-slate-500">
            ({formatFileSize(fileSizeBytes)})
          </span>
        ) : null}
        .
      </p>
      <button
        type="button"
        onClick={onDownload}
        disabled={isDownloading}
        className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-400"
      >
        {isDownloading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Download className="h-4 w-4" />
        )}
        Скачать PDF
      </button>
    </div>
  );
}

function FailedView({
  errorMessage,
  onRetry,
  isRetrying,
}: {
  errorMessage: string | null;
  onRetry: () => void;
  isRetrying: boolean;
}) {
  return (
    <div className="mt-4 rounded-md border border-red-300 bg-red-50 p-4">
      <div className="flex items-start gap-3">
        <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-red-900">
            Генерация завершилась с ошибкой
          </p>
          <p className="mt-1 break-words text-sm text-red-800">
            {errorMessage || "Внутренняя ошибка. Попробуйте ещё раз."}
          </p>
          <button
            type="button"
            onClick={onRetry}
            disabled={isRetrying}
            className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isRetrying ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Попробовать снова
          </button>
        </div>
      </div>
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} МБ`;
}
