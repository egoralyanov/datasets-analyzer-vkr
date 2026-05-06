// Карточка генерации и скачивания PDF-отчёта по анализу. Размещается
// последней в цепочке AnalysisResult — семантически финальный шаг.
//
// Состояния (local):
//   idle       — reportId === null, никто ещё не нажимал «Сгенерировать».
//   generating — reportId !== null, статус ∈ {pending, running}.
//   ready      — статус === success.
//   error      — статус === failed.
//
// Известное ограничение: при перезагрузке страницы компонент стартует в idle
// даже если у анализа уже есть готовый success-отчёт в БД. Бэк не отдаёт
// «последний report для analysis_id». 90% сценариев закрываются через 409 +
// reason="report_in_progress" — берём report_id и подцепляемся к polling.
//
// Стиль (Sprint 5, Phase 2): архивная карточка с hairline-обводкой; кнопки
// в archive-стиле (paper.50 фон, 1px ink.700 обводка, инверсия на hover).
// См. frontend/DESIGN_TOKENS.md.
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import { Loader2 } from "lucide-react";
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
    <div>
      {localError && (
        <div className="mb-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
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
            setReportId(null);
            startMutation.mutate();
          }}
          isRetrying={startMutation.isPending}
        />
      )}
    </div>
  );
}

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
    <div className="space-y-4">
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-600">
        Сводка датасета, флаги качества, распределения, рекомендация и метрики
        baseline в одном PDF. Обычно занимает 15–30 секунд.
      </p>
      <ArchiveButton onClick={onStart} disabled={disabled}>
        {isStarting ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ЗАПУСК…
          </>
        ) : (
          "СГЕНЕРИРОВАТЬ ОТЧЁТ"
        )}
      </ArchiveButton>
      {analysisStatus !== "done" && (
        <p className="font-sans text-xs text-paper-500">
          Кнопка станет активной после завершения анализа.
        </p>
      )}
    </div>
  );
}

function GeneratingView() {
  return (
    <div className="flex items-start gap-3 border-l-[3px] border-info-500 bg-paper-50 px-4 py-3">
      <Loader2 className="mt-0.5 h-4 w-4 animate-spin text-info-500" />
      <div>
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-info-700">
          ГЕНЕРАЦИЯ ОТЧЁТА
        </p>
        <p className="mt-1 font-serif text-sm leading-relaxed text-paper-700">
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
    <div className="space-y-3">
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        Отчёт готов
        {fileSizeBytes !== null && (
          <span className="ml-1 font-mono text-sm text-paper-500">
            ({formatFileSize(fileSizeBytes)})
          </span>
        )}
        .
      </p>
      <ArchiveButton onClick={onDownload} disabled={isDownloading}>
        {isDownloading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            СКАЧИВАНИЕ…
          </>
        ) : (
          "СКАЧАТЬ PDF"
        )}
      </ArchiveButton>
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
    <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-3">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-critical-700">
        FAIL · ГЕНЕРАЦИЯ ЗАВЕРШИЛАСЬ С ОШИБКОЙ
      </p>
      <p className="mt-1 break-words font-serif text-sm leading-relaxed text-paper-700">
        {errorMessage || "Внутренняя ошибка. Попробуйте ещё раз."}
      </p>
      <div className="mt-3">
        <ArchiveButton onClick={onRetry} disabled={isRetrying}>
          {isRetrying ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ПОВТОР…
            </>
          ) : (
            "ПОПРОБОВАТЬ СНОВА"
          )}
        </ArchiveButton>
      </div>
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} МБ`;
}

function ArchiveButton({
  children,
  onClick,
  disabled = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-2 border border-ink-700 bg-paper-50 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-paper-50 disabled:hover:text-ink-700"
    >
      {children}
    </button>
  );
}
