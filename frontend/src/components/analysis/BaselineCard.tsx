// Baseline: 4 состояния (not_started/running/done/failed) + таблица метрик
// и список важности признаков.
//
// Стиль (Sprint 5, Phase 2): таблица — настоящая <table> с border-collapse и
// hairline-руллями, без скруглений; кнопки в archive-стиле — paper.50 фон с
// 1px ink.700-обводкой, на hover инверсия.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4.
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { analysesApi } from "../../api/analyses";
import { useBaselinePolling } from "../../hooks/useBaselinePolling";
import type {
  BaselineResponse,
  BaselineResult,
  BaselineStatus,
  MetricValue,
} from "../../types/analysis";

type Props = {
  analysisId: string;
  taskType: string | undefined;
};

const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  ridge: "Ridge",
};

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  f1_macro: "F1 macro",
  f1_weighted: "F1 weighted",
  roc_auc: "ROC AUC",
  mae: "MAE",
  rmse: "RMSE",
  r2: "R²",
};

function formatMetric(value: MetricValue): string {
  if (!Number.isFinite(value.mean)) return "—";
  return `${value.mean.toFixed(3)} ± ${value.std.toFixed(3)}`;
}

export function BaselineCard({ analysisId, taskType }: Props) {
  const queryClient = useQueryClient();
  const polling = useBaselinePolling(analysisId);
  const [localError, setLocalError] = useState<string | null>(null);

  const startMutation = useMutation({
    mutationFn: () => analysesApi.startBaseline(analysisId),
    onMutate: () => setLocalError(null),
    onSuccess: (resp) => {
      queryClient.setQueryData<BaselineResponse>(["baseline", analysisId], {
        baseline_status: resp.baseline_status,
        baseline: null,
        baseline_error: null,
      });
      queryClient.invalidateQueries({ queryKey: ["baseline", analysisId] });
    },
    onError: (err: unknown) => {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ??
        (err instanceof Error ? err.message : "Не удалось запустить обучение");
      setLocalError(detail);
    },
  });

  const data = polling.data;
  const status: BaselineStatus = data?.baseline_status ?? "not_started";

  return (
    <div>
      {localError && <InlineError message={localError} />}

      {status === "not_started" && (
        <NotStartedView
          taskType={taskType}
          onStart={() => startMutation.mutate()}
          isStarting={startMutation.isPending}
        />
      )}

      {status === "running" && <RunningView />}

      {status === "done" && data?.baseline && (
        <DoneView result={data.baseline} />
      )}

      {status === "failed" && (
        <FailedView
          errorMessage={data?.baseline_error}
          onRetry={() => startMutation.mutate()}
          isRetrying={startMutation.isPending}
        />
      )}
    </div>
  );
}

function NotStartedView({
  taskType,
  onStart,
  isStarting,
}: {
  taskType: string | undefined;
  onStart: () => void;
  isStarting: boolean;
}) {
  const isStub = taskType === "CLUSTERING" || taskType === "NOT_READY";

  return (
    <div className="space-y-4">
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-600">
        Обучим две baseline-модели (линейная + RandomForest) с 5-fold
        кросс-валидацией. Колонки с подозрением на утечку исключаются
        автоматически. Обычно занимает 5–15 секунд.
      </p>

      {isStub && (
        <div className="border-l-[3px] border-info-500 bg-paper-50 px-4 py-3 font-serif text-sm leading-relaxed text-paper-700">
          <span className="font-sans text-xs font-medium uppercase tracking-wider text-info-700">
            ⓘ INFO
          </span>{" "}
          Для текущего типа задачи baseline не обучается — результатом будет
          краткая текстовая рекомендация по алгоритмам.
        </div>
      )}

      <ArchiveButton onClick={onStart} disabled={isStarting}>
        {isStarting ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ЗАПУСК…
          </>
        ) : (
          "ОБУЧИТЬ BASELINE"
        )}
      </ArchiveButton>
    </div>
  );
}

function RunningView() {
  return (
    <div className="flex items-start gap-3 border-l-[3px] border-info-500 bg-paper-50 px-4 py-3">
      <Loader2 className="mt-0.5 h-4 w-4 animate-spin text-info-500" />
      <div>
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-info-700">
          ИДЁТ ОБУЧЕНИЕ
        </p>
        <p className="mt-1 font-serif text-sm leading-relaxed text-paper-700">
          Препроцессинг, кросс-валидация, расчёт важности признаков. Обычно
          5–15 секунд.
        </p>
      </div>
    </div>
  );
}

function FailedView({
  errorMessage,
  onRetry,
  isRetrying,
}: {
  errorMessage: string | null | undefined;
  onRetry: () => void;
  isRetrying: boolean;
}) {
  return (
    <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-3">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-critical-700">
        FAIL · ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ
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

function DoneView({ result }: { result: BaselineResult }) {
  if (result.note && result.models.length === 0) {
    return (
      <div className="border-l-[3px] border-paper-400 bg-paper-100/60 px-4 py-3 font-serif text-sm leading-relaxed text-paper-700">
        {result.note}
      </div>
    );
  }

  const allMetricKeys = Array.from(
    new Set(
      result.models.flatMap((m) => Object.keys(result.metrics[m] ?? {})),
    ),
  );

  const importanceEntries = Object.entries(result.feature_importance).sort(
    (a, b) => b[1] - a[1],
  );
  const maxImportance = importanceEntries[0]?.[1] ?? 1;

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-2">
        <Field label="Строк в обучении" value={result.n_rows_used} />
        <Field label="Признаков использовано" value={result.n_features_used} />
        {result.excluded_columns_due_to_leakage.length > 0 && (
          <Field
            label="Исключено по leakage"
            value={result.excluded_columns_due_to_leakage.join(", ")}
            full
          />
        )}
      </dl>

      <div>
        <SubsectionLabel>Метрики моделей</SubsectionLabel>
        <div className="mt-2 overflow-x-auto border border-paper-300">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-paper-300 bg-paper-100/60 text-left font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
                <th className="px-3 py-2 font-medium">Модель</th>
                {allMetricKeys.map((key) => (
                  <th key={key} className="px-3 py-2 font-medium">
                    {METRIC_LABELS[key] ?? key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.models.map((model) => (
                <tr
                  key={model}
                  className="border-b border-paper-200 align-top last:border-b-0"
                >
                  <td className="px-3 py-2 font-sans text-sm text-paper-800">
                    {MODEL_LABELS[model] ?? model}
                  </td>
                  {allMetricKeys.map((key) => {
                    const value = result.metrics[model]?.[key];
                    return (
                      <td
                        key={key}
                        className="px-3 py-2 font-mono text-xs text-paper-700"
                      >
                        {value ? formatMetric(value) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {importanceEntries.length > 0 && (
        <div>
          <SubsectionLabel>Важность признаков (топ-{importanceEntries.length})</SubsectionLabel>
          <div className="mt-2 overflow-x-auto border border-paper-300">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-paper-300 bg-paper-100/60 text-left font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
                  <th className="w-10 px-3 py-2 font-medium text-right">#</th>
                  <th className="px-3 py-2 font-medium">Признак</th>
                  <th className="px-3 py-2 font-medium text-right">Важность</th>
                  <th className="w-1/3 px-3 py-2 font-medium">Шкала</th>
                </tr>
              </thead>
              <tbody>
                {importanceEntries.map(([name, importance], idx) => {
                  const widthPct = Math.max(
                    2,
                    Math.round((importance / maxImportance) * 100),
                  );
                  return (
                    <tr
                      key={name}
                      className="border-b border-paper-200 last:border-b-0"
                    >
                      <td className="px-3 py-2 text-right font-mono text-xs text-paper-400">
                        {String(idx + 1).padStart(2, "0")}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-paper-800">
                        {name}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-paper-700">
                        {importance.toFixed(3)}
                      </td>
                      <td className="px-3 py-2">
                        <div className="h-1.5 bg-paper-200">
                          <div
                            className="h-full bg-ink-700"
                            style={{ width: `${widthPct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <p className="font-sans text-xs uppercase tracking-wider text-paper-500">
        ОБУЧЕНО:{" "}
        <span className="font-mono normal-case tracking-normal text-paper-700">
          {new Date(result.trained_at).toLocaleString("ru-RU")}
        </span>
      </p>
    </div>
  );
}

function Field({
  label,
  value,
  full = false,
}: {
  label: string;
  value: React.ReactNode;
  full?: boolean;
}) {
  return (
    <div className={full ? "sm:col-span-2" : ""}>
      <dt className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className="mt-0.5 font-mono text-sm text-paper-800">{value}</dd>
    </div>
  );
}

function SubsectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
      {children}
    </h3>
  );
}

function InlineError({ message }: { message: string }) {
  return (
    <div className="mb-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
      {message}
    </div>
  );
}

// Архивная кнопка — paper.50 фон + 1px ink.700 обводка, hover инвертирует.
// Используется в нескольких analysis-компонентах; локальная (не экспортируем
// общий UI-примитив до Phase 3 — пилот не делает раскат шире страницы).
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
