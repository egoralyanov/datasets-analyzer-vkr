import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertOctagon,
  FlaskConical,
  Info,
  Loader2,
  Play,
  RefreshCw,
} from "lucide-react";
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

// Маппинг технических имён моделей и метрик в человеческие подписи.
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

// Не все метрики ограничены [0, 1] — для R² и ошибок (MAE/RMSE) форматируем
// без процентов и не показываем «уверенность» как % шкалу.
const PERCENT_METRICS = new Set([
  "accuracy",
  "precision",
  "recall",
  "f1",
  "f1_macro",
  "f1_weighted",
  "roc_auc",
]);

function formatMetric(value: MetricValue, key: string): string {
  if (!Number.isFinite(value.mean)) return "—";
  if (PERCENT_METRICS.has(key)) {
    return `${value.mean.toFixed(3)} ± ${value.std.toFixed(3)}`;
  }
  // Регрессионные метрики (mae/rmse/r2) — фиксированные 3 знака.
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
      // Оптимистично подменяем кеш: сразу показываем running/done — не ждём
      // следующего polling-тика. Polling быстро подхватит реальное состояние.
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
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <FlaskConical className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">
          Baseline-обучение
        </h2>
      </div>

      {localError && (
        <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {localError}
        </div>
      )}

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
    </section>
  );
}

// =============================================================================
//                               СОСТОЯНИЯ
// =============================================================================

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
    <div className="mt-4 space-y-4">
      <p className="text-sm text-slate-600">
        Обучим две baseline-модели (линейная + RandomForest) с 5-fold
        кросс-валидацией. Колонки с подозрением на утечку исключаются
        автоматически. Обычно занимает 5–15 секунд.
      </p>

      {isStub && (
        <div className="flex items-start gap-2 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
          <Info className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            Для текущего типа задачи baseline не обучается — результатом будет
            краткая текстовая рекомендация по алгоритмам.
          </span>
        </div>
      )}

      <button
        type="button"
        onClick={onStart}
        disabled={isStarting}
        className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-400"
      >
        {isStarting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Play className="h-4 w-4" />
        )}
        Обучить baseline
      </button>
    </div>
  );
}

function RunningView() {
  return (
    <div className="mt-4 flex items-start gap-3 rounded-md border border-blue-200 bg-blue-50 p-4">
      <Loader2 className="mt-0.5 h-5 w-5 animate-spin text-blue-600" />
      <div className="text-sm">
        <p className="font-medium text-blue-900">Идёт обучение</p>
        <p className="mt-1 text-blue-800">
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
    <div className="mt-4 rounded-md border border-red-300 bg-red-50 p-4">
      <div className="flex items-start gap-3">
        <AlertOctagon className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-red-900">
            Обучение завершилось с ошибкой
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

function DoneView({ result }: { result: BaselineResult }) {
  // CLUSTERING / NOT_READY: backend кладёт `note` вместо реальных моделей.
  if (result.note && result.models.length === 0) {
    return (
      <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
        <Info className="mr-2 inline-block h-4 w-4 align-text-bottom text-slate-500" />
        {result.note}
      </div>
    );
  }

  // Универсальная сборка таблицы: строки — модели, столбцы — все встретившиеся
  // метрики (например, BINARY имеет roc_auc, MULTICLASS — нет).
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
    <div className="mt-4 space-y-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-600">
        <span>
          Строк в обучении:{" "}
          <span className="font-mono text-slate-800">{result.n_rows_used}</span>
        </span>
        <span>
          Признаков:{" "}
          <span className="font-mono text-slate-800">
            {result.n_features_used}
          </span>
        </span>
        {result.excluded_columns_due_to_leakage.length > 0 && (
          <span>
            Исключено по leakage:{" "}
            <span className="font-mono text-slate-800">
              {result.excluded_columns_due_to_leakage.join(", ")}
            </span>
          </span>
        )}
      </div>

      <div>
        <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">
          Метрики моделей
        </h3>
        <div className="mt-2 overflow-x-auto rounded-md border border-slate-200">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
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
                  className="border-t border-slate-200 align-top"
                >
                  <td className="px-3 py-2 font-medium text-slate-900">
                    {MODEL_LABELS[model] ?? model}
                  </td>
                  {allMetricKeys.map((key) => {
                    const value = result.metrics[model]?.[key];
                    return (
                      <td
                        key={key}
                        className="px-3 py-2 font-mono text-xs text-slate-700"
                      >
                        {value ? formatMetric(value, key) : "—"}
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
          <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">
            Важность признаков (топ-10)
          </h3>
          <ul className="mt-3 space-y-2">
            {importanceEntries.map(([name, importance]) => {
              const widthPct = Math.max(
                3,
                Math.round((importance / maxImportance) * 100),
              );
              return (
                <li key={name} className="flex items-center gap-3 text-sm">
                  <span
                    className="w-44 shrink-0 truncate text-slate-700"
                    title={name}
                  >
                    {name}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full bg-blue-500"
                      style={{ width: `${widthPct}%` }}
                    />
                  </div>
                  <span className="w-14 text-right font-mono text-xs text-slate-600">
                    {importance.toFixed(3)}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="border-t border-slate-200 pt-3 text-xs text-slate-500">
        Обучено:{" "}
        <span className="font-mono text-slate-700">
          {new Date(result.trained_at).toLocaleString("ru-RU")}
        </span>
      </div>
    </div>
  );
}
