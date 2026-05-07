// Baseline: 4 состояния (not_started/running/done/failed) + таблица метрик
// и список важности признаков.
//
// Sprint 6, Phase 2: компонент стал презентационным — состояние и мутации
// живут в `useBaselineActions` (hooks/), а карточка получает их через props.
// На широких экранах метрики и важность признаков идут side-by-side
// (`lg:grid-cols-[3fr_2fr]`), на узких — стек.
//
// Sprint 6, Phase 3: после таблицы метрик показывается ruled-блок с
// семантическим вердиктом (success / warning / critical). Берём ЛУЧШУЮ
// модель (для регрессии — по r2, для классификации — по f1_macro/f1) и
// классифицируем её качество порогами:
//   r2:        ≥0.7 success | ≥0.3 success | ≥0 warning | <0 critical
//   accuracy:  ≥0.9 success | ≥0.7 success | ≥(1/N + 0.1) warning |
//              около 1/N — critical
// Это локальная фронт-логика, без изменения backend-контракта.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4.
import { Loader2 } from "lucide-react";
import type { BaselineActions } from "../../hooks/useBaselineActions";
import type { BaselineResult, MetricValue } from "../../types/analysis";

type Props = {
  taskType: string | undefined;
  /** Число классов target для классификации (для расчёта random baseline). */
  nClasses?: number;
  actions: BaselineActions;
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

export function BaselineCard({ taskType, nClasses, actions }: Props) {
  const { status, result, pollingError, startError, isStarting, start } =
    actions;

  return (
    <div>
      {startError && <InlineError message={startError} />}

      {status === "not_started" && (
        <NotStartedView
          taskType={taskType}
          onStart={start}
          isStarting={isStarting}
        />
      )}

      {status === "running" && <RunningView />}

      {status === "done" && result && (
        <DoneView result={result} taskType={taskType} nClasses={nClasses} />
      )}

      {status === "failed" && (
        <FailedView
          errorMessage={pollingError}
          onRetry={start}
          isRetrying={isStarting}
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

function DoneView({
  result,
  taskType,
  nClasses,
}: {
  result: BaselineResult;
  taskType: string | undefined;
  nClasses: number | undefined;
}) {
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

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
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
            <SubsectionLabel>
              Важность признаков (топ-{importanceEntries.length})
            </SubsectionLabel>
            <div className="mt-2 overflow-x-auto border border-paper-300">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-paper-300 bg-paper-100/60 text-left font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
                    <th className="w-10 px-3 py-2 text-right font-medium">#</th>
                    <th className="px-3 py-2 font-medium">Признак</th>
                    <th className="px-3 py-2 text-right font-medium">
                      Важность
                    </th>
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
      </div>

      <VerdictBlock result={result} taskType={taskType} nClasses={nClasses} />

      <p className="font-sans text-xs uppercase tracking-wider text-paper-500">
        ОБУЧЕНО:{" "}
        <span className="font-mono normal-case tracking-normal text-paper-700">
          {new Date(result.trained_at).toLocaleString("ru-RU")}
        </span>
      </p>
    </div>
  );
}

// =============================================================================
//                    Семантический вердикт по результатам baseline
// =============================================================================

type VerdictTone = "success" | "warning" | "critical";

type Verdict = {
  tone: VerdictTone;
  text: string;
};

// Цветовая гамма ruled-блока: hairline-граница в семантическом цвете +
// текст в *-700 + лёгкий тинт фона *-50/40, без жёсткой заливки (см.
// DESIGN_TOKENS.md, п.7).
const VERDICT_STYLES: Record<VerdictTone, string> = {
  success: "border-success-500 bg-success-50/40 text-success-700",
  warning: "border-warning-500 bg-warning-50/40 text-warning-700",
  critical: "border-critical-500 bg-critical-50/40 text-critical-700",
};

const VERDICT_LABEL: Record<VerdictTone, string> = {
  success: "ВЕРДИКТ · ХОРОШО",
  warning: "ВЕРДИКТ · СЛАБО",
  critical: "ВЕРДИКТ · НЕТ СИГНАЛА",
};

function VerdictBlock({
  result,
  taskType,
  nClasses,
}: {
  result: BaselineResult;
  taskType: string | undefined;
  nClasses: number | undefined;
}) {
  const verdict = computeVerdict(result, taskType, nClasses);
  if (verdict === null) return null;

  return (
    <div
      className={`border border-l-[3px] px-4 py-3 ${VERDICT_STYLES[verdict.tone]}`}
    >
      <p className="font-sans text-xs font-medium uppercase tracking-wider">
        {VERDICT_LABEL[verdict.tone]}
      </p>
      <p className="mt-1 font-serif text-base leading-relaxed">
        {verdict.text}
      </p>
    </div>
  );
}

function computeVerdict(
  result: BaselineResult,
  taskType: string | undefined,
  nClasses: number | undefined,
): Verdict | null {
  if (result.models.length === 0) return null;

  if (taskType === "REGRESSION") {
    const r2 = bestMetricMean(result, "r2");
    if (r2 === null) return null;
    return regressionVerdict(r2);
  }

  if (
    taskType === "BINARY_CLASSIFICATION" ||
    taskType === "MULTICLASS_CLASSIFICATION"
  ) {
    // Классификация: лучшую модель выбираем по f1_macro (multiclass) или
    // f1 (binary) — что есть; вердикт строится по accuracy этой модели.
    const rankerKey = taskType === "BINARY_CLASSIFICATION" ? "f1" : "f1_macro";
    const bestModel = pickBestModel(result, rankerKey);
    if (bestModel === null) return null;
    const accuracy = result.metrics[bestModel]?.accuracy?.mean;
    if (accuracy === undefined || !Number.isFinite(accuracy)) return null;
    const effectiveClasses =
      taskType === "BINARY_CLASSIFICATION" ? 2 : nClasses ?? 0;
    return classificationVerdict(accuracy, effectiveClasses);
  }

  return null;
}

// Среди всех моделей берём максимальное среднее по указанной метрике.
// Используется для регрессии (r2): лучший результат — лучший вердикт.
function bestMetricMean(
  result: BaselineResult,
  metricKey: string,
): number | null {
  let best: number | null = null;
  for (const model of result.models) {
    const value = result.metrics[model]?.[metricKey]?.mean;
    if (value === undefined || !Number.isFinite(value)) continue;
    if (best === null || value > best) best = value;
  }
  return best;
}

// Для классификации сначала отбираем модель по f1, потом смотрим её accuracy:
// f1 точнее отражает качество при дисбалансе, accuracy — интуитивнее для
// итогового вердикта.
function pickBestModel(
  result: BaselineResult,
  metricKey: string,
): string | null {
  let bestModel: string | null = null;
  let bestValue = -Infinity;
  for (const model of result.models) {
    const value = result.metrics[model]?.[metricKey]?.mean;
    if (value === undefined || !Number.isFinite(value)) continue;
    if (value > bestValue) {
      bestValue = value;
      bestModel = model;
    }
  }
  return bestModel;
}

function regressionVerdict(r2: number): Verdict {
  if (r2 >= 0.7) {
    return {
      tone: "success",
      text:
        "Модель находит сильный сигнал в данных. Признаки хорошо объясняют целевую переменную.",
    };
  }
  if (r2 >= 0.3) {
    return {
      tone: "success",
      text:
        "Модель находит умеренный сигнал. Возможна доработка признаков для улучшения качества.",
    };
  }
  if (r2 >= 0) {
    return {
      tone: "warning",
      text:
        "Сигнал в данных слабый. Признаки слабо объясняют целевую переменную.",
    };
  }
  return {
    tone: "critical",
    text:
      "Модель работает не лучше предсказания среднего. Признаки не несут полезного сигнала для целевой переменной — пересмотрите состав признаков или постановку задачи.",
  };
}

function classificationVerdict(accuracy: number, nClasses: number): Verdict {
  if (accuracy >= 0.9) {
    return {
      tone: "success",
      text: "Модель уверенно различает классы.",
    };
  }
  if (accuracy >= 0.7) {
    return {
      tone: "success",
      text:
        "Модель различает классы, но есть пространство для улучшения.",
    };
  }

  // Ниже 0.7 — сравниваем с базой случайного выбора 1/N. Если N неизвестно
  // (nClasses == 0) — деградируем к простому порогу 0.5 как для бинарной.
  const randomBaseline = nClasses > 0 ? 1 / nClasses : 0.5;

  if (Math.abs(accuracy - randomBaseline) < 0.05) {
    return {
      tone: "critical",
      text:
        "Модель работает на уровне случайного выбора. Признаки не несут полезного сигнала для целевой переменной.",
    };
  }
  if (accuracy >= randomBaseline + 0.1) {
    return {
      tone: "warning",
      text:
        "Модель обходит случайный выбор, но слабо. Для надёжного прогноза требуется доработка.",
    };
  }
  // Между «около base» и «base+0.1» — тоже warning, но с акцентом на разницу.
  return {
    tone: "warning",
    text:
      "Модель обходит случайный выбор, но слабо. Для надёжного прогноза требуется доработка.",
  };
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
