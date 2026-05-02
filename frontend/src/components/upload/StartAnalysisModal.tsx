import { useEffect, useState } from "react";
import { Loader2, X } from "lucide-react";
import type { HintedTaskType } from "../../types/analysis";

const HINTED_OPTIONS: { value: "" | HintedTaskType; label: string }[] = [
  { value: "", label: "Не указывать" },
  { value: "binary_classification", label: "Бинарная классификация" },
  { value: "multiclass_classification", label: "Мультикласс-классификация" },
  { value: "regression", label: "Регрессия" },
  { value: "clustering", label: "Кластеризация (без target)" },
];

type Props = {
  open: boolean;
  columns: string[];
  isPending: boolean;
  errorText?: string | null;
  onClose: () => void;
  onSubmit: (params: {
    target_column: string | null;
    hinted_task_type: HintedTaskType | null;
  }) => void;
};

export function StartAnalysisModal({
  open,
  columns,
  isPending,
  errorText,
  onClose,
  onSubmit,
}: Props) {
  const [target, setTarget] = useState<string>("");
  const [hint, setHint] = useState<"" | HintedTaskType>("");

  // Сброс формы при открытии модалки.
  useEffect(() => {
    if (open) {
      setTarget("");
      setHint("");
    }
  }, [open]);

  // Esc закрывает модалку.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isPending) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, isPending, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={() => !isPending && onClose()}
    >
      <div
        className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <h2 className="text-lg font-semibold text-slate-900">
            Запустить анализ
          </h2>
          <button
            type="button"
            onClick={onClose}
            disabled={isPending}
            className="rounded-md p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-50"
            aria-label="Закрыть"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mt-1 text-sm text-slate-600">
          Параметры опциональны. Если задачи нет (анализ ради профилирования) —
          оставьте оба поля пустыми.
        </p>

        <form
          className="mt-5 space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit({
              target_column: target || null,
              hinted_task_type: hint || null,
            });
          }}
        >
          <div>
            <label
              htmlFor="target-column"
              className="block text-sm font-medium text-slate-700"
            >
              Целевая переменная
            </label>
            <select
              id="target-column"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={isPending}
              className="mt-1 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-60"
            >
              <option value="">Без target</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label
              htmlFor="hinted-task-type"
              className="block text-sm font-medium text-slate-700"
            >
              Подсказка по типу задачи
            </label>
            <select
              id="hinted-task-type"
              value={hint}
              onChange={(e) => setHint(e.target.value as "" | HintedTaskType)}
              disabled={isPending}
              className="mt-1 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-60"
            >
              {HINTED_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          {errorText && (
            <div className="rounded-md border border-red-200 bg-red-50 p-2 text-xs text-red-800">
              {errorText}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isPending}
              className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-60"
            >
              Отмена
            </button>
            <button
              type="submit"
              disabled={isPending}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-60"
            >
              {isPending && <Loader2 className="h-4 w-4 animate-spin" />}
              Запустить
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
