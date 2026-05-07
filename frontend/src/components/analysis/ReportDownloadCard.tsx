// Карточка генерации и скачивания PDF-отчёта по анализу.
//
// Sprint 6, Phase 2: компонент стал презентационным — состояние и мутации
// живут в `useReportActions` (hooks/), карточка получает их через props.
// Это позволяет sticky-bar триггерить генерацию и одновременно карточке
// отображать статус (один источник истины).
//
// Состояния:
//   idle       — reportId === null
//   generating — status ∈ {pending, running}
//   ready      — status === success
//   error      — status === failed
//
// Известное ограничение из Sprint 4: при перезагрузке страницы reportId в
// useReportActions сбрасывается в null даже если success-отчёт уже есть в БД
// (нет latest-report endpoint'а). 90% случаев закрываются через 409 +
// reason="report_in_progress".
import { Loader2 } from "lucide-react";
import type { ReportActions } from "../../hooks/useReportActions";
import type { AnalysisStatus } from "../../types/analysis";

type Props = {
  analysisStatus: AnalysisStatus;
  actions: ReportActions;
};

export function ReportDownloadCard({ analysisStatus, actions }: Props) {
  const {
    reportId,
    status,
    fileSizeBytes,
    pollingError,
    flowError,
    isStarting,
    isDownloading,
    start,
    download,
    retry,
  } = actions;

  return (
    <div>
      {flowError && (
        <div className="mb-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
          {flowError}
        </div>
      )}

      {reportId === null && (
        <IdleView
          onStart={start}
          disabled={analysisStatus !== "done" || isStarting}
          isStarting={isStarting}
          analysisStatus={analysisStatus}
        />
      )}

      {reportId !== null && (status === "pending" || status === "running") && (
        <GeneratingView />
      )}

      {reportId !== null && status === "success" && (
        <ReadyView
          onDownload={download}
          isDownloading={isDownloading}
          fileSizeBytes={fileSizeBytes}
        />
      )}

      {reportId !== null && status === "failed" && (
        <FailedView
          errorMessage={pollingError}
          onRetry={retry}
          isRetrying={isStarting}
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
