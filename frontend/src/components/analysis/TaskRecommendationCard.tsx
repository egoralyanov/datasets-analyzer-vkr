import { useState } from "react";
import { ChevronDown, ChevronRight, HelpCircle, Target } from "lucide-react";
import type {
  RecommendationSource,
  TaskRecommendation,
  TaskTypeCode,
} from "../../types/analysis";

type Props = {
  recommendation: TaskRecommendation | null;
};

const TASK_TYPE_LABEL: Record<TaskTypeCode, string> = {
  BINARY_CLASSIFICATION: "Бинарная классификация",
  MULTICLASS_CLASSIFICATION: "Многоклассовая классификация",
  REGRESSION: "Регрессия",
  CLUSTERING: "Кластеризация",
  NOT_READY: "Данные не готовы для ML",
};

const SOURCE_LABEL: Record<RecommendationSource, string> = {
  rules: "Правила",
  ml: "ML-модель",
  hybrid: "Гибрид",
};

const SOURCE_DESCRIPTION: Record<RecommendationSource, string> = {
  rules: "Решено детерминированными правилами без обращения к ML.",
  ml: "Пограничный случай — решение принято мета-классификатором.",
  hybrid: "Правила задали направление, ML-модель уточнила тип.",
};

// Цветовая шкала уверенности: 90%+ — стабильно, 70-90% — приемлемо,
// меньше 70% — мягкий сигнал «решение неустойчивое».
function confidenceTone(confidence: number): {
  pillBg: string;
  pillText: string;
  bar: string;
} {
  if (confidence >= 0.9) {
    return {
      pillBg: "bg-emerald-50 border-emerald-200",
      pillText: "text-emerald-700",
      bar: "bg-emerald-500",
    };
  }
  if (confidence >= 0.7) {
    return {
      pillBg: "bg-amber-50 border-amber-200",
      pillText: "text-amber-700",
      bar: "bg-amber-500",
    };
  }
  return {
    pillBg: "bg-slate-100 border-slate-200",
    pillText: "text-slate-700",
    bar: "bg-slate-400",
  };
}

export function TaskRecommendationCard({ recommendation }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (recommendation === null) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <div className="flex items-center gap-2">
          <Target className="h-5 w-5 text-blue-600" />
          <h2 className="text-lg font-semibold text-slate-900">Тип задачи</h2>
        </div>
        <div className="mt-4 flex items-start gap-3 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
          <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
          <span>
            Не удалось определить тип задачи. Обычно это значит, что
            мета-классификатор недоступен — попробуйте обновить модель командой
            <code className="mx-1 rounded bg-white px-1 py-0.5 font-mono text-xs text-slate-800">
              make train-meta
            </code>
            и перезапустить анализ.
          </span>
        </div>
      </section>
    );
  }

  const label = TASK_TYPE_LABEL[recommendation.task_type_code];
  const confidencePct = Math.round(recommendation.confidence * 100);
  const tone = confidenceTone(recommendation.confidence);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <Target className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">Тип задачи</h2>
      </div>

      <div className="mt-4 flex flex-wrap items-baseline gap-x-3 gap-y-2">
        <span className="text-2xl font-semibold text-slate-900">{label}</span>
        <span
          className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${tone.pillBg} ${tone.pillText}`}
        >
          уверенность {confidencePct}%
        </span>
        <span
          className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs font-medium text-slate-600"
          title={SOURCE_DESCRIPTION[recommendation.source]}
        >
          {SOURCE_LABEL[recommendation.source]}
        </span>
      </div>

      <div
        className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-100"
        aria-label={`Уверенность ${confidencePct}%`}
      >
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${tone.bar}`}
          style={{ width: `${confidencePct}%` }}
        />
      </div>

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-blue-700 hover:text-blue-900"
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        {expanded ? "Скрыть обоснование" : "Почему такая рекомендация?"}
      </button>

      {expanded && (
        <div className="mt-3 space-y-3 rounded-md border border-slate-200 bg-slate-50 p-4">
          {recommendation.applied_rules.length > 0 && (
            <div>
              <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Применённые правила
              </h3>
              <ul className="mt-2 space-y-2">
                {recommendation.applied_rules.map((rule) => (
                  <li
                    key={rule.code}
                    className="rounded border border-slate-200 bg-white p-3 text-sm text-slate-800"
                  >
                    <span className="mr-2 rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-700">
                      {rule.code}
                    </span>
                    {rule.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {recommendation.explanation && (
            <div>
              <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Объяснение
              </h3>
              <pre className="mt-2 whitespace-pre-wrap rounded border border-slate-200 bg-white p-3 font-sans text-sm text-slate-700">
                {recommendation.explanation}
              </pre>
            </div>
          )}

          {recommendation.ml_probabilities && (
            <div>
              <h3 className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Вероятности классов (ML)
              </h3>
              <ul className="mt-2 space-y-1">
                {Object.entries(recommendation.ml_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <li
                      key={cls}
                      className="flex items-center gap-3 text-sm text-slate-700"
                    >
                      <span className="w-56 truncate font-mono text-xs">
                        {TASK_TYPE_LABEL[cls as TaskTypeCode] ?? cls}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{ width: `${Math.round(prob * 100)}%` }}
                        />
                      </div>
                      <span className="w-10 text-right font-mono text-xs text-slate-600">
                        {Math.round(prob * 100)}%
                      </span>
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
