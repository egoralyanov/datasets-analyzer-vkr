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
// Sprint 6, Phase 3: над списком замечаний добавлена сводная строка
// с breakdown'ом по severity (всего · критичных · предупреждений ·
// информационных) — даёт быструю общую оценку без скролла. В empty-
// case показывается строка «OK · ПРОБЛЕМ НЕ ОБНАРУЖЕНО».
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.2 + принцип «границы вместо
// заливок».
import { useState } from "react";
import { pluralize } from "../../lib/pluralize";
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

// Лейблы для разрядов summary-строки. Critical и warning склоняются
// (1 → ед.ч., 2-4 → мн.ч., 5+ → род.п.), info — короткий неизменяемый
// бейдж «ИНФО». Это даёт более компактную сводку «1 ПРЕДУПРЕЖДЕНИЕ ·
// 3 ИНФО» вместо «1 ПРЕДУПРЕЖДЕНИЙ · 3 ИНФОРМАЦИОННЫХ».
const SUMMARY_LABELS: Record<
  Severity,
  | { kind: "plural"; forms: readonly [string, string, string] }
  | { kind: "fixed"; label: string }
> = {
  critical: {
    kind: "plural",
    forms: ["КРИТИЧНОЕ", "КРИТИЧНЫХ", "КРИТИЧНЫХ"] as const,
  },
  warning: {
    kind: "plural",
    forms: ["ПРЕДУПРЕЖДЕНИЕ", "ПРЕДУПРЕЖДЕНИЯ", "ПРЕДУПРЕЖДЕНИЙ"] as const,
  },
  info: { kind: "fixed", label: "ИНФО" },
};

const TOTAL_FORMS: readonly [string, string, string] = [
  "ЗАМЕЧАНИЕ",
  "ЗАМЕЧАНИЯ",
  "ЗАМЕЧАНИЙ",
] as const;

export function QualityFlags({ flags }: Props) {
  if (flags.length === 0) {
    return (
      <div className="space-y-3">
        <SummaryStrip
          totals={{ critical: 0, warning: 0, info: 0 }}
          totalCount={0}
        />
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

  const totals = {
    critical: grouped.critical.length,
    warning: grouped.warning.length,
    info: grouped.info.length,
  };

  return (
    <div className="space-y-8">
      <SummaryStrip totals={totals} totalCount={flags.length} />
      {SEVERITY_ORDER.map((sev) => {
        const list = grouped[sev];
        if (list.length === 0) return null;
        return (
          <div key={sev}>
            <h3 className="mb-2 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
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

// Сводная строка над списком замечаний. В non-empty case — счётчик «всего»
// + breakdown по severity (критичных / предупреждений / информационных).
// В empty case — успех («OK · ПРОБЛЕМ НЕ ОБНАРУЖЕНО»). Числа в Mono,
// лейблы Sans uppercase, разделители «·» в paper-500 — выдержано в
// archive-тоне (см. DESIGN_TOKENS.md, п. 7).
function SummaryStrip({
  totals,
  totalCount,
}: {
  totals: Record<Severity, number>;
  totalCount: number;
}) {
  if (totalCount === 0) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-y border-paper-300 px-3 py-2 font-sans text-xs uppercase tracking-wider">
        <span className="font-mono text-sm normal-case tracking-normal text-success-700">
          OK
        </span>
        <span className="text-paper-500">·</span>
        <span className="text-success-700">ПРОБЛЕМ НЕ ОБНАРУЖЕНО</span>
      </div>
    );
  }

  const totalLabel = pluralize(totalCount, TOTAL_FORMS);

  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-y border-paper-300 px-3 py-2 font-sans text-xs uppercase tracking-wider">
      <span>
        <span className="font-mono text-sm normal-case tracking-normal text-paper-800">
          {totalCount}
        </span>{" "}
        <span className="text-paper-700">{totalLabel}</span>
      </span>
      {SEVERITY_ORDER.map((sev) => (
        <SummaryItem key={sev} count={totals[sev]} tone={sev} />
      ))}
    </div>
  );
}

function SummaryItem({ count, tone }: { count: number; tone: Severity }) {
  // Разрядное «0 КРИТИЧНЫХ» приглушаем до paper-500, чтобы не привлекать
  // лишнего внимания (нет — значит нет). Ненулевые показываем семантическим
  // цветом — это «маркер на полях», не заливка (DESIGN_TOKENS.md, п. 7).
  // Лейбл выбираем через SUMMARY_LABELS: critical/warning склоняются по
  // числу, info остаётся неизменяемым «ИНФО».
  const config = SUMMARY_LABELS[tone];
  const label =
    config.kind === "plural" ? pluralize(count, config.forms) : config.label;
  const numberColor = count === 0 ? "text-paper-500" : "text-paper-800";
  const labelColor = count === 0 ? "text-paper-500" : SEVERITY_TEXT[tone];
  return (
    <span className="flex items-baseline gap-x-3">
      <span className="text-paper-500">·</span>
      <span>
        <span
          className={`font-mono text-sm normal-case tracking-normal ${numberColor}`}
        >
          {count}
        </span>{" "}
        <span className={labelColor}>{label}</span>
      </span>
    </span>
  );
}

function FlagRow({ flag }: { flag: QualityFlag }) {
  // Sprint 6, Phase 2.5: всегда стартуем с collapsed-state — пользователь
  // сам раскрывает то, что интересно. Ряд при этом сжат до 2 строк:
  // верхняя — pill + code + rule_name + кнопка справа, нижняя — message.
  const [expanded, setExpanded] = useState(false);
  const hasContext = flag.context && Object.keys(flag.context).length > 0;
  const sev = flag.severity;

  return (
    <div
      className={`border-b border-paper-200 last:border-b-0 border-l-[3px] ${SEVERITY_RULE[sev]} ${SEVERITY_TINT[sev]} px-4 py-2`}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <SeverityPill tone={sev}>{SEVERITY_BADGE[sev]}</SeverityPill>
          <span className="font-mono text-xs text-paper-700">
            {flag.rule_code}
          </span>
          <span className="font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
            {flag.rule_name}
          </span>
        </div>

        {hasContext && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="font-sans text-xs font-medium uppercase tracking-wider text-ink-700 underline-offset-2 hover:underline"
          >
            {expanded ? "СКРЫТЬ" : "ПОДРОБНЕЕ →"}
          </button>
        )}
      </div>

      <p className="mt-1 font-serif text-[0.9375rem] leading-snug text-paper-700">
        {flag.message}
      </p>

      {expanded && hasContext && (
        <dl className="mt-2 space-y-1 border-t border-paper-200 pt-2">
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
