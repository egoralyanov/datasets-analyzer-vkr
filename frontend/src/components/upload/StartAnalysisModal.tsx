// Модалка запуска анализа.
//
// Стиль (Sprint 5, Phase 3.11): scientific/archive — overlay с paper-фоном
// и единственной разрешённой `shadow-overlay` (см. DESIGN_TOKENS.md, раздел 5);
// форма в hairline-обводке, инпуты архивные, кнопки в archive-стиле.
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

type Props = {
  open: boolean;
  columns: string[];
  isPending: boolean;
  errorText?: string | null;
  onClose: () => void;
  onSubmit: (params: { target_column: string | null }) => void;
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

  // Сброс формы при открытии модалки.
  useEffect(() => {
    if (open) {
      setTarget("");
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
      className="fixed inset-0 z-50 flex items-center justify-center bg-paper-900/50 p-4"
      onClick={() => !isPending && onClose()}
    >
      <div
        className="w-full max-w-md border border-paper-300 bg-paper-50 p-6 shadow-overlay"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <div>
            <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
              ЗАПУСК
            </p>
            <h2 className="mt-1 font-serif text-[1.375rem] font-semibold leading-snug text-paper-900">
              Запустить анализ
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isPending}
            className="font-sans text-sm text-paper-500 transition-colors hover:text-ink-700 disabled:opacity-50"
            aria-label="Закрыть"
          >
            ✕
          </button>
        </div>
        <p className="mt-2 font-serif text-sm leading-relaxed text-paper-600">
          Если у вас есть колонка-цель для предсказания — укажите её. Если
          нет — система проанализирует структуру данных и предложит подходящие
          типы задач машинного обучения.
        </p>

        <form
          className="mt-5 space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit({
              target_column: target || null,
            });
          }}
        >
          <div>
            <label
              htmlFor="target-column"
              className="block font-sans text-xs font-medium uppercase tracking-wider text-paper-500"
            >
              Целевая переменная (опционально)
            </label>
            <select
              id="target-column"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              disabled={isPending}
              className="mt-1.5 block w-full border border-paper-400 bg-paper-50 px-3 py-2 font-mono text-sm text-paper-800 focus:border-ink-700 focus:outline-none focus:ring-0 disabled:opacity-60"
            >
              <option value="">Без целевой переменной</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {errorText && (
            <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-3 py-2 font-sans text-xs text-critical-700">
              {errorText}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isPending}
              className="border border-paper-400 bg-paper-50 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              ОТМЕНА
            </button>
            <button
              type="submit"
              disabled={isPending}
              className="inline-flex items-center gap-2 border border-ink-700 bg-ink-700 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isPending && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              ЗАПУСТИТЬ
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
