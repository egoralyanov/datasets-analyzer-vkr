// Confirm-модалка удаления датасета (Sprint 6, Phase 5.1).
//
// Открывается с {datasetId, filename}. На mount подгружает usage
// (analyses_count + reports_count) — пока грузится, показывается строка-
// плейсхолдер «загружаем...». Кнопка «УДАЛИТЬ» доступна сразу: пользователь
// не должен ждать счётчики, чтобы удалить (текст про counts появится
// по ходу).
//
// Поведение:
// - Успешный DELETE → закрытие, toast.success, queryClient.invalidate.
// - Ошибка DELETE → inline-error в модалке, окно остаётся открытым,
//   позволяя повторить или отменить.
// - 409 на DELETE датасета невозможен по контракту бэка (нет running-state
//   гейтов), поэтому отдельной обработки 409 здесь нет.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Trash2 } from "lucide-react";

import { datasetsApi } from "../../api/datasets";
import { Modal } from "../ui/Modal";
import { PLURAL_FORMS, pluralize } from "../../lib/pluralize";
import { toast } from "../../lib/toast";

type Props = {
  open: boolean;
  datasetId: string | null;
  filename: string | null;
  onClose: () => void;
  /** Хук для дополнительной реакции (например, очистить current на странице). */
  onDeleted?: (datasetId: string) => void;
};

export function DeleteDatasetModal({
  open,
  datasetId,
  filename,
  onClose,
  onDeleted,
}: Props) {
  const queryClient = useQueryClient();

  const usageQuery = useQuery({
    queryKey: ["dataset-usage", datasetId],
    queryFn: () => datasetsApi.getUsage(datasetId!),
    enabled: open && datasetId !== null,
    // Свежие счётчики при каждом открытии — поведение «удалит N анализов»
    // должно отражать актуальное состояние на момент клика.
    staleTime: 0,
    refetchOnMount: "always",
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => datasetsApi.remove(id),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      onDeleted?.(id);
      onClose();
      toast.success("Датасет удалён");
    },
  });

  const isDeleting = deleteMutation.isPending;
  const errorText = deleteMutation.isError
    ? "Не удалось удалить. Повторите попытку или обратитесь к администратору."
    : null;

  const title = filename ? `Удалить датасет «${filename}»?` : "Удалить датасет?";

  return (
    <Modal
      open={open}
      onClose={() => {
        if (isDeleting) return;
        deleteMutation.reset();
        onClose();
      }}
      title={title}
      subtitle="УДАЛЕНИЕ"
      disableDismiss={isDeleting}
      size="md"
    >
      <UsageBody usage={usageQuery.data} loading={usageQuery.isLoading} />

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
          onClick={() => datasetId && deleteMutation.mutate(datasetId)}
          disabled={isDeleting || datasetId === null}
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

// Тело модалки с описанием последствий. Пока usage не пришёл — показываем
// плейсхолдер «Считаем связанные артефакты…». 0/0 не выводится до
// загрузки данных, чтобы не вводить пользователя в заблуждение.
function UsageBody({
  usage,
  loading,
}: {
  usage: { analyses_count: number; reports_count: number } | undefined;
  loading: boolean;
}) {
  if (loading || usage === undefined) {
    return (
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-600">
        Считаем связанные артефакты…
      </p>
    );
  }

  const { analyses_count, reports_count } = usage;
  if (analyses_count === 0 && reports_count === 0) {
    return (
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        Анализов и отчётов нет. Действие необратимо.
      </p>
    );
  }

  const analysesWord = pluralize(analyses_count, PLURAL_FORMS.analysis);
  const reportsWord = pluralize(reports_count, PLURAL_FORMS.report);
  return (
    <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
      Также будет удалено:{" "}
      <span className="font-mono text-[0.875rem]">{analyses_count}</span>{" "}
      {analysesWord},{" "}
      <span className="font-mono text-[0.875rem]">{reports_count}</span>{" "}
      {reportsWord}. Действие необратимо.
    </p>
  );
}
