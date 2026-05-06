// Сводка датасета — «specimen header» в духе выходных данных научной
// публикации: definition-list с лейблами в Sans uppercase tracking и
// значениями в Mono, hairline-разделители между строк.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4.
import type { MetaFeatures } from "../../types/analysis";
import { formatNumber } from "../../lib/format";

type Props = {
  meta: MetaFeatures;
  filename?: string;
  targetColumn: string | null;
};

export function DatasetSummary({ meta, filename, targetColumn }: Props) {
  const totalMissingPct = (meta.total_missing_pct ?? 0) * 100;
  const duplicatesPct = (meta.duplicate_rows_pct ?? 0) * 100;
  const sampling = meta.sampling;

  // Перечень dtype'ов в формате «int: 5, float: 2, object: 4».
  const dtypeBreakdown = Object.entries(meta.dtype_counts || {})
    .map(([dtype, count]) => `${dtype} ${count}`)
    .join(" · ");

  // Тип target отображаем словом, не enum-кодом.
  const targetKindLabel = (() => {
    if (!meta.target_kind) return null;
    return meta.target_kind === "categorical" ? "категориальная" : "числовая";
  })();

  return (
    <div className="border border-paper-300 bg-paper-50">
      <dl className="divide-y divide-paper-200">
        {filename && <Row label="Имя файла" value={filename} mono />}
        <Row
          label="Строк × столбцов"
          value={`${formatNumber(meta.n_rows)} × ${formatNumber(meta.n_cols)}`}
          mono
        />
        <Row
          label="Размер в памяти"
          value={`${(meta.memory_mb ?? 0).toFixed(2)} МБ`}
          mono
        />
        {dtypeBreakdown && (
          <Row label="Типы колонок" value={dtypeBreakdown} mono />
        )}
        <Row
          label="Пропусков (всего)"
          value={`${totalMissingPct.toFixed(2)}%`}
          mono
          tone={totalMissingPct > 10 ? "warning" : undefined}
        />
        <Row
          label="Дубликатов строк"
          value={`${duplicatesPct.toFixed(2)}%`}
          mono
          tone={duplicatesPct > 5 ? "warning" : undefined}
        />
        {targetColumn && (
          <Row
            label="Целевая переменная"
            value={
              <>
                <span className="font-mono">«{targetColumn}»</span>
                {targetKindLabel && (
                  <span className="ml-2 text-paper-500">
                    — {targetKindLabel}
                  </span>
                )}
              </>
            }
          />
        )}
        {sampling?.sampled && (
          <Row
            label="Сэмплирование"
            value={
              <>
                <span className="font-mono">
                  {formatNumber(sampling.sample_size)}
                </span>
                <span className="text-paper-500"> из </span>
                <span className="font-mono">
                  {formatNumber(sampling.original_size)}
                </span>
                <span className="text-paper-500">
                  {" "}
                  · стратифицировано по target
                </span>
              </>
            }
            tone="info"
          />
        )}
      </dl>
    </div>
  );
}

type Tone = "warning" | "info";

function Row({
  label,
  value,
  mono = false,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
  tone?: Tone;
}) {
  const valueClass = [
    mono ? "font-mono" : "font-sans",
    tone === "warning" && "text-warning-700",
    tone === "info" && "text-info-700",
    !tone && "text-paper-800",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="grid grid-cols-[12rem_1fr] gap-6 px-5 py-3">
      <dt className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className={`text-sm ${valueClass}`}>{value}</dd>
    </div>
  );
}
