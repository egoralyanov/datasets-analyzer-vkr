// Флаги качества данных — ruled-строки с 3px-полосой severity слева и
// hairline-разделителем снизу. Группировка по severity, expandable
// definition-list с context'ом в Mono.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.2.
import { useState } from "react";
import type { QualityFlag, Severity } from "../../types/analysis";

type Props = {
  flags: QualityFlag[];
};

const SEVERITY_ORDER: Severity[] = ["critical", "warning", "info"];

const SEVERITY_LABEL: Record<Severity, string> = {
  critical: "Критические",
  warning: "Предупреждения",
  info: "Информация",
};

const SEVERITY_BADGE: Record<Severity, string> = {
  critical: "CRIT",
  warning: "WARN",
  info: "INFO",
};

// Цвет рульки слева и текста бейджа. Тинт фона строки даём только для
// critical — в остальных случаях фон остаётся paper.50, чтобы текст не
// «терялся» на цветном поле (см. п. 7 манифеста — semantic-цвета как
// отметки, а не заливки).
const SEVERITY_RULE: Record<Severity, string> = {
  critical: "border-critical-500",
  warning: "border-warning-500",
  info: "border-info-500",
};

const SEVERITY_TEXT: Record<Severity, string> = {
  critical: "text-critical-700",
  warning: "text-warning-700",
  info: "text-info-700",
};

const SEVERITY_TINT: Record<Severity, string> = {
  critical: "bg-critical-50/70",
  warning: "bg-paper-50",
  info: "bg-paper-50",
};

export function QualityFlags({ flags }: Props) {
  if (flags.length === 0) {
    return (
      <div className="border-l-[3px] border-success-500 bg-paper-50 px-5 py-4">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-success-700">
          ОК · ПРОБЛЕМ НЕ ОБНАРУЖЕНО
        </p>
        <p className="mt-1 font-serif text-[0.9375rem] text-paper-700">
          Ни одно из 12 правил качества не сработало. Датасет пригоден к
          обучению.
        </p>
      </div>
    );
  }

  const grouped: Record<Severity, QualityFlag[]> = {
    critical: [],
    warning: [],
    info: [],
  };
  for (const f of flags) {
    grouped[f.severity].push(f);
  }

  return (
    <div className="space-y-8">
      {SEVERITY_ORDER.map((sev) => {
        const list = grouped[sev];
        if (list.length === 0) return null;
        return (
          <div key={sev}>
            <h3 className="mb-3 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
              <span className={SEVERITY_TEXT[sev]}>{SEVERITY_LABEL[sev]}</span>
              <span className="ml-2 font-mono normal-case tracking-normal text-paper-500">
                ({list.length})
              </span>
            </h3>
            <div className="border-y border-paper-200">
              {list.map((flag, idx) => (
                <FlagRow key={`${sev}-${idx}`} flag={flag} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function FlagRow({ flag }: { flag: QualityFlag }) {
  const [expanded, setExpanded] = useState(false);
  const hasContext = flag.context && Object.keys(flag.context).length > 0;
  const sev = flag.severity;

  return (
    <div
      className={`border-b border-paper-200 last:border-b-0 border-l-[3px] ${SEVERITY_RULE[sev]} ${SEVERITY_TINT[sev]} px-5 py-3`}
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span
          className={`font-mono text-xs font-medium ${SEVERITY_TEXT[sev]}`}
          aria-label={`severity: ${sev}`}
        >
          {SEVERITY_BADGE[sev]}
        </span>
        <span className="font-mono text-xs text-paper-700">
          {flag.rule_code}
        </span>
        <span className="font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
          {flag.rule_name}
        </span>
      </div>

      <p className="mt-2 font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        {flag.message}
      </p>

      {hasContext && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 underline-offset-2 hover:underline"
        >
          {expanded ? "СКРЫТЬ ДЕТАЛИ" : "ПОДРОБНЕЕ →"}
        </button>
      )}

      {expanded && hasContext && (
        <dl className="mt-3 grid grid-cols-1 gap-x-8 gap-y-1.5 border-t border-paper-200 pt-3 sm:grid-cols-2">
          {Object.entries(flag.context!).map(([key, value]) => (
            <div
              key={key}
              className="flex justify-between gap-3 border-b border-dotted border-paper-200 pb-1 last:border-b-0"
            >
              <dt className="font-sans text-xs uppercase tracking-wider text-paper-500">
                {key}
              </dt>
              <dd className="font-mono text-xs text-paper-800">
                {formatContextValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function formatContextValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
