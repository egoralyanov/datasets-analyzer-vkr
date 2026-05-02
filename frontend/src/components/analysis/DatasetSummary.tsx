import { Database, FileText, Info } from "lucide-react";
import type { MetaFeatures } from "../../types/analysis";
import { formatNumber } from "../../lib/format";

type Props = {
  meta: MetaFeatures;
  filename?: string;
  targetColumn: string | null;
};

const DTYPE_PILL_COLORS: Record<string, string> = {
  int: "bg-blue-50 text-blue-700 border-blue-200",
  float: "bg-blue-50 text-blue-700 border-blue-200",
  str: "bg-emerald-50 text-emerald-700 border-emerald-200",
  object: "bg-emerald-50 text-emerald-700 border-emerald-200",
  bool: "bg-slate-100 text-slate-700 border-slate-200",
  datetime: "bg-violet-50 text-violet-700 border-violet-200",
};

function pillColorFor(dtype: string): string {
  for (const key of Object.keys(DTYPE_PILL_COLORS)) {
    if (dtype.toLowerCase().includes(key)) return DTYPE_PILL_COLORS[key];
  }
  return "bg-slate-100 text-slate-700 border-slate-200";
}

export function DatasetSummary({ meta, filename, targetColumn }: Props) {
  const totalMissingPct = (meta.total_missing_pct ?? 0) * 100;
  const sampling = meta.sampling;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <Database className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">Сводка датасета</h2>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <Stat label="Строк" value={formatNumber(meta.n_rows)} />
        <Stat label="Столбцов" value={formatNumber(meta.n_cols)} />
        <Stat
          label="Размер в памяти"
          value={`${(meta.memory_mb ?? 0).toFixed(2)} МБ`}
        />
      </div>

      {filename && (
        <p className="mt-4 flex items-center gap-1.5 text-sm text-slate-600">
          <FileText className="h-4 w-4" />
          <span className="truncate">{filename}</span>
        </p>
      )}

      {targetColumn && (
        <p className="mt-2 text-sm text-slate-600">
          Целевая переменная:{" "}
          <span className="font-medium text-slate-900">«{targetColumn}»</span>
          {meta.target_kind && (
            <span className="ml-2 text-slate-500">
              (
              {meta.target_kind === "categorical"
                ? "категориальная"
                : "числовая"}
              )
            </span>
          )}
        </p>
      )}

      <div className="mt-4">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Типы колонок
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {Object.entries(meta.dtype_counts || {}).map(([dtype, count]) => (
            <span
              key={dtype}
              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${pillColorFor(
                dtype,
              )}`}
            >
              {dtype}
              <span className="rounded bg-white/60 px-1 text-[10px]">
                {count}
              </span>
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Stat
          label="Общая доля пропусков"
          value={`${totalMissingPct.toFixed(1)}%`}
          highlight={totalMissingPct > 10}
        />
        <Stat
          label="Доля дубликатов"
          value={`${((meta.duplicate_rows_pct ?? 0) * 100).toFixed(1)}%`}
          highlight={(meta.duplicate_rows_pct ?? 0) > 0.05}
        />
      </div>

      {sampling?.sampled && (
        <div className="mt-4 flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <Info className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            Использован сэмпл из{" "}
            <strong>{formatNumber(sampling.sample_size)}</strong> строк
            (исходный размер{" "}
            <strong>{formatNumber(sampling.original_size)}</strong>).
            Сэмплирование стратифицированное по target — требование
            производительности при размере свыше 50 000 строк.
          </span>
        </div>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p
        className={`mt-1 text-2xl font-semibold ${
          highlight ? "text-amber-700" : "text-slate-900"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
