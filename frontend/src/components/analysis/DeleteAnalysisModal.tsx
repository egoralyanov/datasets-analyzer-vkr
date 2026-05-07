// Confirm-модалка удаления анализа (Sprint 6, Phase 5.2).
//
// Открывается с {analysisId, datasetFilename, taskTypeCode, completedAt} и
// сама вызывает DELETE /api/analyses/{id}. Различает 409-исключения по
// тексту detail (см. backend/app/api/analyses.py, delete_analysis):
//   - "Cannot delete analysis while profiling is in progress" → toast.warning
//     "Нельзя удалить пока идёт профайлинг анализа", модалка закрывается.
//   - "Cannot delete analysis while baseline training is in progress" →
//     toast.warning "Нельзя удалить пока обучается базовая модель",
//     модалка закрывается.
//   - Прочие ошибки (500/network/etc) → inline-error в модалке, окно
//     остаётся открытым — пользователь может повторить или отменить.
//
// Контекстная развязка через onDeleted callback: после успешного DELETE
// модалка инвалидирует react-query кэш ['history'] и ['analyses-latest'],
// показывает toast.success, закрывает себя — а парент сам решает, нужно
// ли куда-то перейти. На /analyses/{id} парент дополнительно делает
// removeQueries по ключам ['analysis', id] и ['analysisResult', id]
// перед navigate, чтобы не было stale-рефетча на удалённый id.
//
// completedAt=null означает «анализ ещё бежит» — в теле модалки фраза
// меняется с «от {date}» на «запущен {date}». taskTypeCode=null означает
// «тип ещё не определён» — в этом случае скобки с лейблом убираются
// (см. lib/taskTypes.ts, getTaskTypeLabel).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { Loader2, Trash2 } from "lucide-react";

import { analysesApi } from "../../api/analyses";
import { Modal } from "../ui/Modal";
import { getTaskTypeLabel } from "../../lib/taskTypes";
import { toast } from "../../lib/toast";
import type { TaskTypeCode } from "../../types/analysis";

type Props = {
  open: boolean;
  onClose: () => void;
  analysisId: string | null;
  datasetFilename: string | null;
  taskTypeCode: TaskTypeCode | string | null | undefined;
  /** ISO-строка или Date завершения анализа. null — анализ ещё running,
   *  тогда для подписи берётся startedAt. */
  completedAt: string | Date | null;
  /** ISO-строка или Date запуска анализа. Используется для подписи
   *  «запущен …», когда completedAt=null. */
  startedAt: string | Date;
  /** Хук для парента: успех DELETE — например, navigate('/history'). */
  onDeleted?: () => void;
};

export function DeleteAnalysisModal({
  open,
  onClose,
  analysisId,
  datasetFilename,
  taskTypeCode,
  completedAt,
  startedAt,
  onDeleted,
}: Props) {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => analysesApi.remove(id),
    onSuccess: (_data, id) => {
      // Стираем кэш конкретного анализа, чтобы не словить stale-рефетч на
      // удалённый id при возврате назад в браузере.
      queryClient.removeQueries({ queryKey: ["analysis", id] });
      queryClient.removeQueries({ queryKey: ["analysisResult", id] });
      // Списки обновляем предикатом — затрагиваем все вариации
      // ['history', page, statusFilter] и ['analyses-latest'].
      queryClient.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === "history" || q.queryKey[0] === "analyses-latest",
      });
      toast.success("Анализ удалён");
      deleteMutation.reset();
      onClose();
      onDeleted?.();
    },
    onError: (err) => {
      const detail = extractDetail(err);
      if (detail && detail.includes("profiling")) {
        toast.warning("Нельзя удалить пока идёт профайлинг анализа");
        deleteMutation.reset();
        onClose();
        return;
      }
      if (detail && detail.includes("baseline training")) {
        toast.warning("Нельзя удалить пока обучается базовая модель");
        deleteMutation.reset();
        onClose();
        return;
      }
      // Прочие ошибки — оставляем модалку открытой и показываем inline-error.
      // (deleteMutation.isError остаётся true, errorText ниже отрендерит блок.)
    },
  });

  const isDeleting = deleteMutation.isPending;
  const isInlineError =
    deleteMutation.isError && !isHandledStatusError(deleteMutation.error);
  const errorText = isInlineError
    ? "Не удалось удалить. Повторите попытку или обратитесь к администратору."
    : null;

  const taskLabel = getTaskTypeLabel(taskTypeCode);
  const dateLabel = formatDateLabel(completedAt, startedAt);
  const filename = datasetFilename ?? "—";

  return (
    <Modal
      open={open}
      onClose={() => {
        if (isDeleting) return;
        deleteMutation.reset();
        onClose();
      }}
      title="Удалить анализ?"
      subtitle="УДАЛЕНИЕ"
      disableDismiss={isDeleting}
      size="md"
    >
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        Анализ «{filename}»
        {taskLabel && (
          <>
            {" "}
            (<span className="text-paper-800">{taskLabel}</span>)
          </>
        )}
        , {dateLabel}. PDF-отчёт также будет удалён. Действие необратимо.
      </p>

      {errorText && (
        <div className="mt-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-3 py-2 font-sans text-xs text-critical-700">
          {errorText}
        </div>
      )}

      <div className="mt-5 flex justify-end gap-2">
        <button
          type="button"
          onClick={() => {
            if (isDeleting) return;
            deleteMutation.reset();
            onClose();
          }}
          disabled={isDeleting}
          className="border border-paper-400 bg-paper-50 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          ОТМЕНА
        </button>
        <button
          type="button"
          onClick={() => analysisId && deleteMutation.mutate(analysisId)}
          disabled={isDeleting || analysisId === null}
          className="inline-flex items-center gap-2 border border-critical-500 bg-critical-500 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:border-critical-700 hover:bg-critical-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isDeleting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
          {isDeleting ? "УДАЛЯЕМ…" : "УДАЛИТЬ"}
        </button>
      </div>
    </Modal>
  );
}

// Извлекает {detail: string} из axios-ошибки. На сетевых ошибках вернёт
// undefined — тогда отрабатывает ветка inline-error.
function extractDetail(err: unknown): string | undefined {
  if (err instanceof AxiosError) {
    const data = err.response?.data;
    if (data && typeof data === "object" && "detail" in data) {
      const d = (data as { detail?: unknown }).detail;
      if (typeof d === "string") return d;
    }
  }
  return undefined;
}

// 409 с известным detail обработана toast'ом и НЕ должна показываться
// inline'ом (модалка к этому моменту уже закрыта). Остальное — inline.
function isHandledStatusError(err: unknown): boolean {
  const detail = extractDetail(err);
  if (!detail) return false;
  return detail.includes("profiling") || detail.includes("baseline training");
}

// "от 07.05.2026, 14:32" если completedAt задан, "запущен …" если null
// (тогда дата берётся из startedAt). Формат — toLocaleString("ru-RU") как
// в шапке анализа и в /history.
function formatDateLabel(
  completedAt: string | Date | null,
  startedAt: string | Date,
): string {
  if (completedAt === null) {
    return `запущен ${formatDateTime(toDate(startedAt))}`;
  }
  return `от ${formatDateTime(toDate(completedAt))}`;
}

function toDate(v: string | Date): Date {
  return typeof v === "string" ? new Date(v) : v;
}

function formatDateTime(d: Date): string {
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("ru-RU");
}
