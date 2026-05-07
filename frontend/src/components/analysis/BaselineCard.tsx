// Baseline: 4 состояния (not_started/running/done/failed) + таблица метрик
// и список важности признаков.
//
// Sprint 6, Phase 2: компонент стал презентационным — состояние и мутации
// живут в `useBaselineActions` (hooks/), а карточка получает их через props.
// На широких экранах метрики и важность признаков идут side-by-side
// (`lg:grid-cols-[3fr_2fr]`), на узких — стек.
//
// Sprint 6, Phase 3: после таблицы метрик показывается ruled-блок с
// семантическим вердиктом (4 категории: success / info / warning /
// critical). Берём ЛУЧШУЮ модель (для регрессии — по r2, для
// классификации — по f1_macro/f1) и классифицируем её качество порогами:
//   r2:        ≥0.7 success | ≥0.3 info | ≥0 warning | <0 critical
//   accuracy:  ≥0.9 success | ≥0.7 info | ≥(1/N + 0.1) warning |
//              ≈1/N (|acc-base|<0.05) critical
// Лейблы: УВЕРЕННЫЙ СИГНАЛ / УМЕРЕННЫЙ / СЛАБЫЙ / НЕТ СИГНАЛА.
//
// Дополнительно: если для классификации две модели близки по f1_macro
// (|Δ|<0.02), но сильно расходятся по accuracy (вторая выше на >0.15) —
// это типичная картина при дисбалансе классов (модель угадывает
// доминирующий класс). Тон вердикта override → warning, к тексту
// добавляется уточнение про вторую модель и отсылка к разделу
// «Качество данных».
//
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

type VerdictTone = "success" | "info" | "warning" | "critical";

type Verdict = {
  tone: VerdictTone;
  text: string;
};

// Цветовая гамма ruled-блока: hairline-граница в семантическом цвете +
// текст в *-700 + лёгкий тинт фона *-50/40, без жёсткой заливки (см.
// DESIGN_TOKENS.md, п.7).
const VERDICT_STYLES: Record<VerdictTone, string> = {
  success: "border-success-500 bg-success-50/40 text-success-700",
  info: "border-info-500 bg-info-50/40 text-info-700",
  warning: "border-warning-500 bg-warning-50/40 text-warning-700",
  critical: "border-critical-500 bg-critical-50/40 text-critical-700",
};

const VERDICT_LABEL: Record<VerdictTone, string> = {
  success: "ВЕРДИКТ · УВЕРЕННЫЙ СИГНАЛ",
  info: "ВЕРДИКТ · УМЕРЕННЫЙ",
  warning: "ВЕРДИКТ · СЛАБЫЙ",
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

    const baseVerdict = classificationVerdict(accuracy, effectiveClasses);

    // Override: близкие f1, но сильно расходящийся accuracy → дисбаланс
    // классов; тон → warning, к тексту добавляем фразу про вторую модель.
    const divergence = detectF1AccuracyDivergence(
      result,
      rankerKey,
      bestModel,
      accuracy,
    );
    if (divergence !== null) {
      return {
        tone: "warning",
        text: `${baseVerdict.text} ${divergence}`,
      };
    }
    return baseVerdict;
  }

  return null;
}

// Поиск второй модели, у которой f1 близок к лучшему (|Δ|<0.02), но
// accuracy выше более чем на 0.15. Возвращает уточняющую фразу для
// вердикта или null, если такой модели нет.
function detectF1AccuracyDivergence(
  result: BaselineResult,
  rankerKey: string,
  bestModel: string,
  bestAccuracy: number,
): string | null {
  const bestF1 = result.metrics[bestModel]?.[rankerKey]?.mean;
  if (bestF1 === undefined || !Number.isFinite(bestF1)) return null;

  let candidate: { model: string; accuracy: number } | null = null;
  for (const model of result.models) {
    if (model === bestModel) continue;
    const f1 = result.metrics[model]?.[rankerKey]?.mean;
    const acc = result.metrics[model]?.accuracy?.mean;
    if (f1 === undefined || acc === undefined) continue;
    if (!Number.isFinite(f1) || !Number.isFinite(acc)) continue;

    const f1Close = Math.abs(bestF1 - f1) < 0.02;
    const accMuchHigher = acc - bestAccuracy > 0.15;
    if (f1Close && accMuchHigher) {
      if (candidate === null || acc > candidate.accuracy) {
        candidate = { model, accuracy: acc };
      }
    }
  }

  if (candidate === null) return null;

  const modelLabel = MODEL_LABELS[candidate.model] ?? candidate.model;
  const f1Label = rankerKey === "f1_macro" ? "f1_macro" : "f1";
  return (
    `Однако ${modelLabel} даёт более высокий accuracy ` +
    `(${candidate.accuracy.toFixed(3)}), при близком ${f1Label}. ` +
    "Это типичная картина при дисбалансе классов: модель угадывает " +
    "доминирующий класс, но плохо предсказывает малочисленные. Перед " +
    "выбором модели изучите предупреждения в разделе «Качество данных»."
  );
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
        "Модель объясняет большую часть дисперсии целевой переменной. Качество подходит для прикладного использования.",
    };
  }
  if (r2 >= 0.3) {
    return {
      tone: "info",
      text:
        "Модель улавливает значимую часть зависимости в данных. Точность можно повысить добавлением признаков, нелинейностей и регуляризацией.",
    };
  }
  if (r2 >= 0) {
    return {
      tone: "warning",
      text:
        "Модель работает чуть лучше прогноза средним. Сигнал слабый — возможно, ключевые предикторы отсутствуют в выборке.",
    };
  }
  return {
    tone: "critical",
    text:
      "Модель работает хуже прогноза средним. Признаки шумные либо неинформативны для целевой переменной.",
  };
}

function classificationVerdict(accuracy: number, nClasses: number): Verdict {
  if (accuracy >= 0.9) {
    return {
      tone: "success",
      text:
        "Модель уверенно справляется с задачей. Качество достаточно для прикладного использования при условии корректной валидации на отложенной выборке.",
    };
  }
  if (accuracy >= 0.7) {
    return {
      tone: "info",
      text:
        "Модель распознаёт классы с приемлемой точностью. Для дальнейшего роста качества стоит попробовать настройку гиперпараметров, добавление признаков и более сложные алгоритмы.",
    };
  }

  // Ниже 0.7 — сравниваем с базой случайного выбора 1/N. Если N неизвестно
  // (nClasses == 0) — деградируем к простому порогу 0.5 как для бинарной.
  const randomBaseline = nClasses > 0 ? 1 / nClasses : 0.5;

  if (Math.abs(accuracy - randomBaseline) < 0.05) {
    return {
      tone: "critical",
      text:
        "Модель работает на уровне случайного угадывания. Признаки не несут полезной информации о целевой переменной либо данных недостаточно.",
    };
  }
  // Между «base+0.1» и 0.7, а также между «≈ base» и «base+0.1» — оба
  // случая семантически означают «модель опережает случайное угадывание,
  // но качество низкое». Текст один, тон warning.
  return {
    tone: "warning",
    text:
      "Модель опережает случайное угадывание, но качество низкое. Стоит проверить корректность разметки, баланс классов и информативность признаков.",
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
