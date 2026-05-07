// Действия на странице анализа — sticky-bar внизу экрана (lg+) и
// inline-bar в конце страницы (на узких экранах).
//
// Sprint 6, Phase 2: shortcut-кнопки для длинной страницы. НЕ заменяют
// кнопки в самих карточках BaselineCard / ReportDownloadCard — это удобство
// для пользователя, чтобы не скроллить вниз для запуска или скачивания.
//
// Состояние и мутации приходят через props (см. useBaselineActions /
// useReportActions). Лейблы кнопок и disabled-state мирорят состояние:
//   baseline:  not_started → ЗАПУСТИТЬ; running → ОБУЧАЕТСЯ; done → ОБУЧЕНА;
//              failed → ПОВТОРИТЬ
//   report:    null → СГЕНЕРИРОВАТЬ; pending/running → ГЕНЕРИРУЕТСЯ;
//              success → СКАЧАТЬ PDF; failed → ПОВТОРИТЬ
//
// На узких экранах (<lg) sticky-режим скрыт; для этого случая ниже —
// inline-вариант, который вкладывается в конец страницы.
import { Loader2 } from "lucide-react";
import type { BaselineActions } from "../../hooks/useBaselineActions";
import type { ReportActions } from "../../hooks/useReportActions";
import type { AnalysisStatus } from "../../types/analysis";

type Props = {
  filename: string;
  analysisStatus: AnalysisStatus;
  baseline: BaselineActions;
  report: ReportActions;
};

type ButtonProps = {
  label: string;
  onClick: () => void;
  disabled: boolean;
  busy?: boolean;
};

function buildBaselineButton(
  baseline: BaselineActions,
  analysisStatus: AnalysisStatus,
): ButtonProps {
  const analysisNotDone = analysisStatus !== "done";
  if (analysisNotDone) {
    return {
      label: "БАЗОВАЯ МОДЕЛЬ",
      onClick: () => {},
      disabled: true,
    };
  }
  if (baseline.isStarting) {
    return {
      label: "ЗАПУСК…",
      onClick: () => {},
      disabled: true,
      busy: true,
    };
  }
  switch (baseline.status) {
    case "not_started":
      return {
        label: "ЗАПУСТИТЬ БАЗОВУЮ МОДЕЛЬ",
        onClick: baseline.start,
        disabled: false,
      };
    case "running":
      return {
        label: "БАЗОВАЯ ОБУЧАЕТСЯ…",
        onClick: () => {},
        disabled: true,
        busy: true,
      };
    case "done":
      return {
        label: "БАЗОВАЯ ОБУЧЕНА ✓",
        onClick: () => {},
        disabled: true,
      };
    case "failed":
      return {
        label: "ПОВТОРИТЬ ОБУЧЕНИЕ",
        onClick: baseline.start,
        disabled: false,
      };
  }
}

function buildReportButton(
  report: ReportActions,
  analysisStatus: AnalysisStatus,
): ButtonProps {
  const analysisNotDone = analysisStatus !== "done";
  if (analysisNotDone) {
    return {
      label: "PDF-ОТЧЁТ",
      onClick: () => {},
      disabled: true,
    };
  }
  if (report.reportId === null) {
    if (report.isStarting) {
      return {
        label: "ЗАПУСК…",
        onClick: () => {},
        disabled: true,
        busy: true,
      };
    }
    return {
      label: "СФОРМИРОВАТЬ ОТЧЁТ",
      onClick: report.start,
      disabled: false,
    };
  }
  switch (report.status) {
    case "pending":
    case "running":
      return {
        label: "ОТЧЁТ ГЕНЕРИРУЕТСЯ…",
        onClick: () => {},
        disabled: true,
        busy: true,
      };
    case "success":
      if (report.isDownloading) {
        return {
          label: "СКАЧИВАНИЕ…",
          onClick: () => {},
          disabled: true,
          busy: true,
        };
      }
      return {
        label: "СКАЧАТЬ PDF ↓",
        onClick: report.download,
        disabled: false,
      };
    case "failed":
      return {
        label: "ПОВТОРИТЬ ГЕНЕРАЦИЮ",
        onClick: report.retry,
        disabled: false,
      };
    default:
      return {
        label: "СФОРМИРОВАТЬ ОТЧЁТ",
        onClick: report.start,
        disabled: false,
      };
  }
}

export function StickyActionBar({
  filename,
  analysisStatus,
  baseline,
  report,
}: Props) {
  const baselineBtn = buildBaselineButton(baseline, analysisStatus);
  const reportBtn = buildReportButton(report, analysisStatus);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 hidden border-t border-paper-300 bg-paper-50 lg:block">
      <div className="mx-auto flex max-w-[1200px] items-center justify-between gap-6 px-8 py-3 lg:px-16">
        <div className="flex min-w-0 items-baseline gap-3">
          <span className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
            АНАЛИЗ
          </span>
          <span
            className="truncate font-mono text-sm text-paper-700"
            title={filename}
          >
            {filename}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <BarButton {...baselineBtn} />
          <BarButton {...reportBtn} primary />
        </div>
      </div>
    </div>
  );
}

// Inline-версия для viewport <lg. Вкладывается в конец страницы.
export function InlineActionBar({
  analysisStatus,
  baseline,
  report,
}: Omit<Props, "filename">) {
  const baselineBtn = buildBaselineButton(baseline, analysisStatus);
  const reportBtn = buildReportButton(report, analysisStatus);

  return (
    <div className="mt-12 border-t border-paper-300 pt-5 lg:hidden">
      <p className="mb-3 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        ДЕЙСТВИЯ
      </p>
      <div className="flex flex-wrap gap-2">
        <BarButton {...baselineBtn} />
        <BarButton {...reportBtn} primary />
      </div>
    </div>
  );
}

function BarButton({
  label,
  onClick,
  disabled,
  busy = false,
  primary = false,
}: ButtonProps & { primary?: boolean }) {
  const cls = primary
    ? "border border-ink-700 bg-ink-700 text-paper-50 hover:bg-ink-800"
    : "border border-ink-700 bg-paper-50 text-ink-700 hover:bg-ink-700 hover:text-paper-50";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-2 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${cls}`}
    >
      {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {label}
    </button>
  );
}
