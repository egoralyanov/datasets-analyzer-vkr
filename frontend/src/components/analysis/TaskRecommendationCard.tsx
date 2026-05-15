// Рекомендация типа ML-задачи — hero-блок страницы анализа.
//
// Sprint 6, Phase 2.5: переезд в начало страницы как первая значимая
// секция. Layout — крупный pull-quote с 4px ink-руллёй слева, под ним
// confidence/source в Mono, развёрнутые «Применённые правила» и
// «Объяснение» (по умолчанию видны — для archive-эстетики предпочитаем
// открытую структуру), expandable «Вероятности классов (ML)» — нижний
// уровень детализации, по умолчанию свёрнут.
//
// Sprint 6, Phase 3: между «Применённые правила» и кнопкой «Показать
// вероятности ML» добавлено always-visible пояснение о том, как работает
// Слой 2 (по каким признакам и что именно сравнивается с эталонами).
// Показывается только при source ∈ {hybrid, ml}, потому что при чистых
// правилах Слой 2 не вызывается. См. .project_docs/methods/recommender-ml.md.
//
// См. frontend/DESIGN_TOKENS.md, разделы 7 и 8.4.
import { useState } from "react";
import type {
  RecommendationSource,
  TaskRecommendation,
  TaskTypeCode,
} from "../../types/analysis";
import { TASK_TYPE_LABEL } from "../../lib/taskTypes";

type Props = {
  recommendation: TaskRecommendation | null;
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
  const [mlExpanded, setMlExpanded] = useState(false);

  if (recommendation === null) {
    return (
      <div className="border-l-[4px] border-info-500 pl-6">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-info-700">
          ⓘ ТИП ЗАДАЧИ НЕ ОПРЕДЕЛЁН
        </p>
        <p className="mt-2 font-serif text-[1.0625rem] leading-relaxed text-paper-700">
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
    <div className="border-l-[4px] border-ink-700 pl-6 lg:pl-10">
      {/* Hero pull-quote: тип задачи и confidence/source. */}
      <p className="font-serif text-[2.5rem] font-bold leading-[1.05] tracking-tight text-ink-900 sm:text-[3rem] lg:text-[3.5rem]">
        {label}
      </p>
      <div className="mt-4 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
        <span className="text-paper-500">
          CONFIDENCE{" "}
          <span
            className={`ml-1 font-mono normal-case tracking-normal ${tone}`}
          >
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

      {/* Объяснение — открыто по умолчанию (archive-эстетика любит явность). */}
      {recommendation.explanation && (
        <div className="mt-6 max-w-3xl">
          <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
            Объяснение
          </h3>
          <p className="mt-1.5 whitespace-pre-wrap font-serif text-[1.0625rem] leading-relaxed text-paper-700">
            {recommendation.explanation}
          </p>
        </div>
      )}

      {/* Применённые правила — открыты по умолчанию. */}
      {recommendation.applied_rules.length > 0 && (
        <div className="mt-6 max-w-3xl">
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

      {/* Описание работы Слоя 2 — always-visible, только при hybrid/ml. */}
      {(recommendation.source === "hybrid" ||
        recommendation.source === "ml") && (
        <p className="mt-6 max-w-3xl border-l-2 border-ink-300 bg-paper-100/60 px-4 py-3 font-sans text-sm leading-relaxed text-paper-700">
          Модель Слоя 2 оценивает датасет по 16 мета-признакам — типу целевой
          переменной, кардинальности, балансу классов, корреляциям,
          размерности. Решение основано на близости к похожим эталонам из
          обучающей выборки.
        </p>
      )}

      {/* ML-вероятности — глубокий уровень детализации, expand-able. */}
      {recommendation.ml_probabilities && (
        <div className="mt-6 max-w-3xl">
          <button
            type="button"
            onClick={() => setMlExpanded((v) => !v)}
            aria-expanded={mlExpanded}
            className="font-sans text-xs font-medium uppercase tracking-wider text-ink-700 underline-offset-2 hover:underline"
          >
            {mlExpanded
              ? "СКРЫТЬ ВЕРОЯТНОСТИ ML"
              : "ПОКАЗАТЬ ВЕРОЯТНОСТИ ML →"}
          </button>
          {mlExpanded && (
            <ul className="mt-3 divide-y divide-paper-200 border-y border-paper-200">
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
          )}
        </div>
      )}
    </div>
  );
}
