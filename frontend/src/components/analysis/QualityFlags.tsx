import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Info,
  XCircle,
} from "lucide-react";
import type { QualityFlag, Severity } from "../../types/analysis";

type Props = {
  flags: QualityFlag[];
};

const SEVERITY_ORDER: Severity[] = ["critical", "warning", "info"];

const SEVERITY_LABELS: Record<Severity, string> = {
  critical: "Критические проблемы",
  warning: "Предупреждения",
  info: "Информация",
};

const SEVERITY_STYLES: Record<
  Severity,
  { border: string; bg: string; text: string; icon: typeof XCircle }
> = {
  critical: {
    border: "border-red-300",
    bg: "bg-red-50",
    text: "text-red-900",
    icon: XCircle,
  },
  warning: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    text: "text-amber-900",
    icon: AlertTriangle,
  },
  info: {
    border: "border-blue-300",
    bg: "bg-blue-50",
    text: "text-blue-900",
    icon: Info,
  },
};

export function QualityFlags({ flags }: Props) {
  if (flags.length === 0) {
    return (
      <section className="rounded-lg border border-emerald-300 bg-emerald-50 p-6">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="h-6 w-6 text-emerald-600" />
          <div>
            <h2 className="text-lg font-semibold text-emerald-900">
              Проблем не обнаружено
            </h2>
            <p className="mt-1 text-sm text-emerald-800">
              Ни одно из 12 правил качества не сработало. Датасет готов к
              использованию.
            </p>
          </div>
        </div>
      </section>
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
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <h2 className="text-lg font-semibold text-slate-900">Качество данных</h2>
      <p className="mt-1 text-sm text-slate-600">
        Сработавших правил: <strong>{flags.length}</strong>
      </p>

      <div className="mt-4 space-y-6">
        {SEVERITY_ORDER.map((sev) => {
          const list = grouped[sev];
          if (list.length === 0) return null;
          return (
            <div key={sev}>
              <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">
                {SEVERITY_LABELS[sev]} ({list.length})
              </h3>
              <div className="mt-2 space-y-2">
                {list.map((flag, idx) => (
                  <FlagCard key={`${sev}-${idx}`} flag={flag} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function FlagCard({ flag }: { flag: QualityFlag }) {
  const [expanded, setExpanded] = useState(false);
  const styles = SEVERITY_STYLES[flag.severity];
  const Icon = styles.icon;
  const hasContext = flag.context && Object.keys(flag.context).length > 0;

  return (
    <div className={`rounded-md border ${styles.border} ${styles.bg} p-4`}>
      <div className="flex items-start gap-3">
        <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${styles.text}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span
              className={`rounded bg-white/80 px-1.5 py-0.5 text-xs font-mono font-medium ${styles.text}`}
            >
              {flag.rule_code}
            </span>
            <span className="text-xs text-slate-500">{flag.rule_name}</span>
          </div>
          <p className={`mt-1.5 text-sm ${styles.text}`}>{flag.message}</p>

          {hasContext && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className={`mt-2 inline-flex items-center gap-1 text-xs font-medium ${styles.text} hover:underline`}
            >
              {expanded ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              {expanded ? "Скрыть детали" : "Подробнее"}
            </button>
          )}

          {expanded && hasContext && (
            <dl className="mt-2 grid grid-cols-1 gap-x-4 gap-y-1 rounded border border-white/60 bg-white/60 p-2 text-xs sm:grid-cols-2">
              {Object.entries(flag.context!).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-2">
                  <dt className="font-medium text-slate-700">{key}:</dt>
                  <dd className="font-mono text-slate-900">
                    {formatContextValue(value)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      </div>
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
