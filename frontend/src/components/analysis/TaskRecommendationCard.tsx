// Рекомендация типа ML-задачи — «editorial pull-quote»: крупный
// серифный текст с 3px-рулькой ink.700 слева, под цитатой — мелкие
// технические лейблы (confidence, source) и список применённых правил.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4.
import { useState } from "react";
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
  rules: "rules",
  ml: "ml",
  hybrid: "hybrid",
};

const SOURCE_DESCRIPTION: Record<RecommendationSource, string> = {
  rules: "Решено детерминированными правилами без обращения к ML.",
  ml: "Пограничный случай — решение принято мета-классификатором.",
  hybrid: "Правила задали направление, ML-модель уточнила тип.",
};

function confidenceTone(confidence: number): string {
  if (confidence >= 0.9) return "text-success-700";
  if (confidence >= 0.7) return "text-warning-700";
  return "text-paper-500";
}

export function TaskRecommendationCard({ recommendation }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (recommendation === null) {
    return (
      <div className="border-l-[3px] border-info-500 bg-paper-50 px-5 py-4">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-info-700">
          ⓘ ТИП ЗАДАЧИ НЕ ОПРЕДЕЛЁН
        </p>
        <p className="mt-1 font-serif text-[0.9375rem] leading-relaxed text-paper-700">
          Не удалось определить тип задачи — обычно это значит, что
          мета-классификатор недоступен. Попробуйте обновить модель командой{" "}
          <code className="font-mono text-sm text-paper-800">
            make train-meta
          </code>{" "}
          и перезапустить анализ.
        </p>
      </div>
    );
  }

  const label = TASK_TYPE_LABEL[recommendation.task_type_code];
  const confidencePct = Math.round(recommendation.confidence * 100);
  const tone = confidenceTone(recommendation.confidence);

  return (
    <div>
      <blockquote className="border-l-[3px] border-ink-700 pl-6">
        <p className="font-serif text-[2rem] font-semibold leading-tight tracking-tight text-ink-900">
          {label}
        </p>
        <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
          <span className="text-paper-500">
            CONFIDENCE{" "}
            <span className={`ml-1 font-mono normal-case tracking-normal ${tone}`}>
              {recommendation.confidence.toFixed(3)}
            </span>
            <span className="ml-1 font-mono normal-case tracking-normal text-paper-500">
              ({confidencePct}%)
            </span>
          </span>
          <span
            className="text-paper-500"
            title={SOURCE_DESCRIPTION[recommendation.source]}
          >
            SOURCE{" "}
            <span className="ml-1 font-mono normal-case tracking-normal text-ink-700">
              {SOURCE_LABEL[recommendation.source]}
            </span>
          </span>
        </div>
      </blockquote>

      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="mt-5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 underline-offset-2 hover:underline"
      >
        {expanded ? "СКРЫТЬ ОБОСНОВАНИЕ" : "ПОЧЕМУ ТАКАЯ РЕКОМЕНДАЦИЯ →"}
      </button>

      {expanded && (
        <div className="mt-4 space-y-5 border-t border-paper-200 pt-4">
          {recommendation.applied_rules.length > 0 && (
            <div>
              <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
                Применённые правила
              </h3>
              <ul className="mt-2 divide-y divide-paper-200 border-y border-paper-200">
                {recommendation.applied_rules.map((rule) => (
                  <li
                    key={rule.code}
                    className="flex flex-col gap-1 px-1 py-2 sm:flex-row sm:items-baseline sm:gap-4"
                  >
                    <span className="font-mono text-xs text-ink-700 sm:w-56 sm:shrink-0">
                      {rule.code}
                    </span>
                    <span className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
                      {rule.description}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {recommendation.explanation && (
            <div>
              <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
                Объяснение
              </h3>
              <pre className="mt-2 whitespace-pre-wrap border-l-[3px] border-paper-300 bg-paper-100/60 px-4 py-3 font-serif text-[0.9375rem] leading-relaxed text-paper-700">
                {recommendation.explanation}
              </pre>
            </div>
          )}

          {recommendation.ml_probabilities && (
            <div>
              <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
                Вероятности классов (ML-слой)
              </h3>
              <ul className="mt-2 divide-y divide-paper-200 border-y border-paper-200">
                {Object.entries(recommendation.ml_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <li
                      key={cls}
                      className="grid grid-cols-[1fr_auto] items-center gap-4 px-1 py-2"
                    >
                      <span className="font-sans text-sm text-paper-700">
                        {TASK_TYPE_LABEL[cls as TaskTypeCode] ?? cls}
                      </span>
                      <span className="font-mono text-xs text-paper-800">
                        {prob.toFixed(3)}{" "}
                        <span className="text-paper-500">
                          ({Math.round(prob * 100)}%)
                        </span>
                      </span>
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
