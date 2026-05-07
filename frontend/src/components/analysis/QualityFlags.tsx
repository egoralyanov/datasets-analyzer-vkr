// Флаги качества данных — ruled-строки с 3px-полосой severity слева и
// hairline-разделителем снизу. Группировка по severity, expandable
// definition-list с context'ом в Mono.
//
// Стиль (Sprint 6, Phase 1): severity-бейджи теперь «пилюли» с тонкой
// обводкой (без bg-fill) — CRIT/WARN/INFO/OK. Длинные context-значения
// (например, value_counts с 30+ ключами) больше не переполняют ячейку:
// для значений > 80 символов ряд переключается в block-mode с скролл-
// окном `max-h-32 overflow-y-auto` (компактно для типичного случая,
// читабельно для редкого).
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.2 + принцип «границы вместо
// заливок».
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

// Цвет рульки слева. Тинт фона строки даём только для critical, для
// остальных фон остаётся paper.50, чтобы текст не «терялся» (см. п. 7
// манифеста — semantic как отметки, не заливки).
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

// Тонкая обводка severity-пилюли (без bg-fill — DESIGN_TOKENS.md, п.7).
const SEVERITY_PILL_BORDER: Record<Severity, string> = {
  critical: "border-critical-500",
  warning: "border-warning-500",
  info: "border-info-500",
};

// Порог, за которым context-значение разворачивается в скролл-блок.
const LONG_VALUE_THRESHOLD = 80;

export function QualityFlags({ flags }: Props) {
  if (flags.length === 0) {
    return (
      <div className="border-l-[3px] border-success-500 bg-paper-50 px-5 py-4">
        <SeverityPill tone="success">OK</SeverityPill>
        <p className="mt-2 font-sans text-xs font-medium uppercase tracking-wider text-success-700">
          ПРОБЛЕМ НЕ ОБНАРУЖЕНО
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
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <SeverityPill tone={sev}>{SEVERITY_BADGE[sev]}</SeverityPill>
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
        <dl className="mt-3 space-y-1.5 border-t border-paper-200 pt-3">
          {Object.entries(flag.context!).map(([key, value]) => (
            <ContextEntry key={key} entryKey={key} value={value} />
          ))}
        </dl>
      )}
    </div>
  );
}

// Одна пара ключ/значение в раскрытом context'е. Для коротких значений —
// inline (key слева, value справа). Для длинных (≥ 80 символов, типично
// сериализованный value_counts) — block-mode: key сверху, value снизу
// в скролл-окне max-h-32, чтобы не растягивать строку и не ломать grid.
function ContextEntry({
  entryKey,
  value,
}: {
  entryKey: string;
  value: unknown;
}) {
  const formatted = formatContextValue(value);
  const isLong = formatted.length > LONG_VALUE_THRESHOLD;

  if (isLong) {
    return (
      <div className="border-b border-dotted border-paper-200 pb-1 last:border-b-0">
        <dt className="font-sans text-xs uppercase tracking-wider text-paper-500">
          {entryKey}
        </dt>
        <dd className="mt-1 max-h-32 overflow-y-auto break-all border border-paper-200 bg-paper-50 px-2 py-1 font-mono text-xs leading-relaxed text-paper-800">
          {formatted}
        </dd>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 border-b border-dotted border-paper-200 pb-1 last:border-b-0">
      <dt className="font-sans text-xs uppercase tracking-wider text-paper-500">
        {entryKey}
      </dt>
      <dd className="break-all font-mono text-xs text-paper-800">
        {formatted}
      </dd>
    </div>
  );
}

// Severity-пилюля с тонкой обводкой, без bg-fill. Используется и для
// бейджей замечаний (CRIT/WARN/INFO), и для empty-state (OK).
function SeverityPill({
  tone,
  children,
}: {
  tone: Severity | "success";
  children: React.ReactNode;
}) {
  const cls =
    tone === "success"
      ? "border-success-500 text-success-700"
      : `${SEVERITY_PILL_BORDER[tone]} ${SEVERITY_TEXT[tone]}`;
  return (
    <span
      className={`inline-flex items-center border px-2 py-0.5 font-mono text-[0.6875rem] font-medium tracking-wider ${cls}`}
    >
      {children}
    </span>
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
