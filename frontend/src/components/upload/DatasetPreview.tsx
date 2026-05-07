// Превью датасета после загрузки/открытия: специмен-метаданных, колонки и
// dtypes как нумерованный каталог, таблица первых строк в строгом стиле.
//
// Стиль (Sprint 5, Phase 3.6): scientific/archive — никаких скруглений,
// никаких теней, hairline-таблицы, mono для числовых значений.
import type { DatasetWithPreview } from "../../types/dataset";
import { formatBytes, formatDateTime, formatNumber } from "../../lib/format";

interface Props {
  data: DatasetWithPreview;
  onDelete: () => void;
  onAnalyze: () => void;
}

function dtypeTone(dtype: string): string {
  const t = dtype.toLowerCase();
  if (t.includes("int") || t.includes("float") || t.includes("number")) {
    return "text-info-700";
  }
  if (t.includes("bool")) return "text-paper-600";
  if (t.includes("date") || t.includes("time")) return "text-warning-700";
  return "text-success-700";
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

export function DatasetPreview({ data, onDelete, onAnalyze }: Props) {
  return (
    <div className="border border-paper-300 bg-paper-50">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-paper-200 px-5 py-4">
        <div className="min-w-0 flex-1">
          <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
            ВЫБРАННЫЙ ФАЙЛ
          </p>
          <h3 className="mt-1 truncate font-serif text-[1.25rem] font-semibold leading-snug text-paper-900">
            {data.original_filename}
          </h3>
        </div>
        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={onAnalyze}
            className="inline-flex items-center gap-2 border border-ink-700 bg-ink-700 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            ▶ ЗАПУСТИТЬ АНАЛИЗ
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="inline-flex items-center gap-1.5 border border-paper-400 bg-paper-50 px-3 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-critical-500 hover:text-critical-700"
          >
            УДАЛИТЬ
          </button>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-x-8 gap-y-3 border-b border-paper-200 px-5 py-4 sm:grid-cols-4">
        <Meta label="Формат" value={data.format.toUpperCase()} />
        <Meta label="Размер" value={formatBytes(data.file_size_bytes)} />
        <Meta
          label="Размерность"
          value={
            data.n_rows !== null && data.n_cols !== null
              ? `${formatNumber(data.n_rows)} × ${formatNumber(data.n_cols)}`
              : "—"
          }
        />
        <Meta label="Загружено" value={formatDateTime(data.uploaded_at)} />
      </dl>

      <div className="border-b border-paper-200 px-5 py-4">
        <h4 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
          КОЛОНКИ И ТИПЫ ({data.preview.columns.length})
        </h4>
        <div className="mt-3 grid gap-x-8 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
          {data.preview.columns.map((col, idx) => {
            const dtype = data.preview.dtypes[col] ?? "?";
            return (
              <div
                key={col}
                className="grid grid-cols-[2rem_1fr_auto] items-baseline gap-2 border-b border-dotted border-paper-200 py-1 last:border-b-0"
              >
                <span className="font-mono text-xs text-paper-400">
                  {String(idx + 1).padStart(2, "0")}
                </span>
                <span className="truncate font-mono text-sm text-paper-800">
                  {col}
                </span>
                <span
                  className={`font-mono text-xs ${dtypeTone(dtype)}`}
                  title={dtype}
                >
                  {dtype}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="px-5 py-4">
        <h4 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
          ПРЕВЬЮ ПЕРВЫХ СТРОК ({data.preview.rows.length})
        </h4>
        <div className="mt-3 max-h-96 overflow-auto border border-paper-300">
          <table className="min-w-full border-collapse text-sm">
            <thead className="sticky top-0 bg-paper-100/70 text-left">
              <tr>
                {data.preview.columns.map((col) => (
                  <th
                    key={col}
                    className="whitespace-nowrap border-b border-paper-300 px-3 py-2 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.preview.rows.map((row, i) => (
                <tr
                  key={i}
                  className="border-b border-paper-200 last:border-b-0"
                >
                  {row.map((cell, j) => {
                    const text = renderCell(cell);
                    return (
                      <td
                        key={j}
                        title={text}
                        className="max-w-xs truncate whitespace-nowrap px-3 py-1.5 font-mono text-xs text-paper-800"
                      >
                        {text}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className="mt-0.5 break-all font-mono text-sm text-paper-800">
        {value}
      </dd>
    </div>
  );
}
