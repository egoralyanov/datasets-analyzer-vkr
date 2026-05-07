// Сводка датасета — компактный блок 5 карточек + горизонтальный bar chart
// типов колонок.
//
// Sprint 6, Phase 2: переход от длинной definition-list к dashboard-layout
// научной публикации. 5 stat-карточек (Строки / Столбцы / Размер / Пропуски /
// Дубликаты) — крупные цифры в Plex Serif, лейблы uppercase tracking. Типы
// колонок — пропорциональный CSS-сегментный bar (без Plotly, без SVG —
// просто flexbox с шириной по доле). Цвета согласованы с DatasetPreview.
//
// Имя файла и target в этом блоке больше не показываем — они в шапке страницы.
//
// См. frontend/DESIGN_TOKENS.md, разделы 8.4 и 7 (manifesto).
import type { MetaFeatures } from "../../types/analysis";
import { formatNumber } from "../../lib/format";

type Props = {
  meta: MetaFeatures;
};

type Tone = "warning" | "info" | undefined;

export function DatasetSummary({ meta }: Props) {
  const totalMissingPct = (meta.total_missing_pct ?? 0) * 100;
  const duplicatesPct = (meta.duplicate_rows_pct ?? 0) * 100;
  const sampling = meta.sampling;
  const dtypeEntries = Object.entries(meta.dtype_counts || {});
  const totalCols = dtypeEntries.reduce((sum, [, c]) => sum + c, 0) || 1;

  return (
    <div className="space-y-6">
      <StatStrip
        cells={[
          { label: "Строк", value: formatNumber(meta.n_rows) },
          { label: "Столбцов", value: formatNumber(meta.n_cols) },
          {
            label: "Размер в памяти",
            value: `${(meta.memory_mb ?? 0).toFixed(2)} МБ`,
          },
          {
            label: "Пропусков",
            value: `${totalMissingPct.toFixed(2)}%`,
            tone: totalMissingPct > 10 ? "warning" : undefined,
          },
          {
            label: "Дубликатов",
            value: `${duplicatesPct.toFixed(2)}%`,
            tone: duplicatesPct > 5 ? "warning" : undefined,
          },
        ]}
      />

      {dtypeEntries.length > 0 && (
        <DtypeBar entries={dtypeEntries} totalCols={totalCols} />
      )}

      {sampling?.sampled && (
        <div className="border-l-[3px] border-info-500 bg-paper-50 px-4 py-3">
          <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-info-700">
            ⓘ СЭМПЛИРОВАНИЕ
          </p>
          <p className="mt-1 font-serif text-sm leading-relaxed text-paper-700">
            Использован сэмпл из{" "}
            <span className="font-mono">
              {formatNumber(sampling.sample_size)}
            </span>{" "}
            строк (исходный размер{" "}
            <span className="font-mono">
              {formatNumber(sampling.original_size)}
            </span>
            ); стратифицировано по target.
          </p>
        </div>
      )}
    </div>
  );
}

function StatStrip({
  cells,
}: {
  cells: { label: string; value: string; tone?: Tone }[];
}) {
  // grid с paper-300 фоном и 1px gap создаёт hairline-разделители между
  // карточками без необходимости border на каждой ячейке.
  return (
    <div className="grid grid-cols-2 gap-px border border-paper-300 bg-paper-300 sm:grid-cols-3 lg:grid-cols-5">
      {cells.map((cell) => (
        <StatCell key={cell.label} {...cell} />
      ))}
    </div>
  );
}

function StatCell({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: Tone;
}) {
  const valueClass =
    tone === "warning"
      ? "text-warning-700"
      : tone === "info"
        ? "text-info-700"
        : "text-paper-900";
  return (
    <div className="bg-paper-50 px-4 py-4">
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <p
        className={`mt-1.5 font-serif text-[1.75rem] font-bold leading-none tracking-tight ${valueClass}`}
      >
        {value}
      </p>
    </div>
  );
}

// Цвет dtype-сегмента. Совмещён с цветовой схемой DatasetPreview:
// numeric→info, categorical→success, datetime→warning, bool→neutral.
function dtypeColor(dtype: string): string {
  const t = dtype.toLowerCase();
  if (t.includes("int") || t.includes("float") || t.includes("number")) {
    return "bg-info-500";
  }
  if (t.includes("date") || t.includes("time")) return "bg-warning-500";
  if (t.includes("bool")) return "bg-paper-500";
  if (t.includes("object") || t.includes("string") || t.includes("category")) {
    return "bg-success-500";
  }
  return "bg-paper-400";
}

function DtypeBar({
  entries,
  totalCols,
}: {
  entries: [string, number][];
  totalCols: number;
}) {
  return (
    <div>
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        ТИПЫ КОЛОНОК
      </p>
      <div className="mt-2 flex h-3 w-full overflow-hidden border border-paper-300">
        {entries.map(([dtype, count], idx) => (
          <div
            key={dtype}
            className={`${dtypeColor(dtype)} ${idx > 0 ? "border-l border-paper-50" : ""}`}
            style={{ width: `${(count / totalCols) * 100}%` }}
            title={`${dtype}: ${count}`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5">
        {entries.map(([dtype, count]) => (
          <div key={dtype} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className={`inline-block h-2.5 w-3 ${dtypeColor(dtype)}`}
            />
            <span className="font-mono text-xs text-paper-700">{dtype}</span>
            <span className="font-mono text-xs text-paper-500">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
