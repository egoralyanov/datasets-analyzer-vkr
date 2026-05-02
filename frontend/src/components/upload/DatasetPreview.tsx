import { FileSpreadsheet, FileText, Trash2 } from "lucide-react";
import type { DatasetWithPreview } from "../../types/dataset";
import { formatBytes, formatDateTime, formatNumber } from "../../lib/format";

interface Props {
  data: DatasetWithPreview;
  onDelete: () => void;
}

// Цвет pill-badge по dtype колонки.
function dtypeBadgeClasses(dtype: string): string {
  const t = dtype.toLowerCase();
  if (t.includes("int") || t.includes("float") || t.includes("number")) {
    return "bg-blue-100 text-blue-800";
  }
  if (t.includes("bool")) return "bg-slate-200 text-slate-800";
  if (t.includes("date") || t.includes("time")) {
    return "bg-purple-100 text-purple-800";
  }
  // object / string / category и прочее
  return "bg-green-100 text-green-800";
}

function FormatIcon({ format }: { format: "csv" | "xlsx" }) {
  return format === "xlsx" ? (
    <FileSpreadsheet className="h-5 w-5 text-emerald-600" />
  ) : (
    <FileText className="h-5 w-5 text-blue-600" />
  );
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

export function DatasetPreview({ data, onDelete }: Props) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between gap-3 p-5 border-b border-slate-200">
        <div className="flex items-center gap-3 min-w-0">
          <FormatIcon format={data.format} />
          <h2 className="truncate text-lg font-semibold text-slate-900">
            {data.original_filename}
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            disabled
            title="Появится в Спринте 2"
            className="inline-flex items-center rounded-md bg-slate-300 px-3 py-1.5 text-sm font-medium text-slate-600 cursor-not-allowed"
          >
            Перейти к анализу (скоро)
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="inline-flex items-center gap-1 rounded-md border border-red-200 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-50"
          >
            <Trash2 className="h-4 w-4" />
            Удалить
          </button>
        </div>
      </div>

      <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-5 border-b border-slate-200">
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

      <div className="p-5 border-b border-slate-200">
        <h3 className="text-sm font-semibold text-slate-900">
          Колонки и типы
        </h3>
        <div className="mt-3 flex flex-wrap gap-2">
          {data.preview.columns.map((col) => (
            <span
              key={col}
              className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${dtypeBadgeClasses(
                data.preview.dtypes[col] ?? "",
              )}`}
            >
              {col}
              <span className="opacity-70">
                · {data.preview.dtypes[col] ?? "?"}
              </span>
            </span>
          ))}
        </div>
      </div>

      <div className="p-5">
        <h3 className="text-sm font-semibold text-slate-900">
          Превью первых строк ({data.preview.rows.length})
        </h3>
        <div className="mt-3 max-h-96 overflow-auto rounded-md border border-slate-200">
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-slate-50 text-left">
              <tr>
                {data.preview.columns.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2 font-medium text-slate-700 whitespace-nowrap border-b border-slate-200"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.preview.rows.map((row, i) => (
                <tr key={i} className="border-b border-slate-100 last:border-0">
                  {row.map((cell, j) => {
                    const text = renderCell(cell);
                    return (
                      <td
                        key={j}
                        title={text}
                        className="px-3 py-1.5 text-slate-900 whitespace-nowrap max-w-xs truncate"
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
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-slate-900 break-all">
        {value}
      </dd>
    </div>
  );
}
