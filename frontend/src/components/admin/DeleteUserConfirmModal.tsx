// Подтверждение удаления пользователя (Sprint 6, Phase 5.4).
//
// Открывается из AdminUserDetailModal по клику «УДАЛИТЬ ПОЛЬЗОВАТЕЛЯ».
// Сама вызывает DELETE /api/admin/users/{id} (см. backend/app/api/admin.py,
// delete_user) и обрабатывает три исхода:
//   - 204 → invalidate ['admin', 'users']/['admin', 'stats'], removeQueries
//     для ['admin', 'user', userId], toast.success, onClose + onDeleted.
//   - 409 «your own admin account» → toast.warning «Нельзя удалить
//     собственный аккаунт администратора», onClose. AdminUserDetailModal
//     остаётся открытой — admin может закрыть её сам, либо посмотреть
//     детали ещё раз.
//   - 409 «last admin account» → toast.warning «Нельзя удалить
//     последнего администратора в системе», onClose.
//   - Прочие ошибки → inline-error внутри модалки, окно остаётся
//     открытым (тот же паттерн что в DeleteDataset/Analysis).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { Loader2, Trash2 } from "lucide-react";

import { adminApi } from "../../api/admin";
import { Modal } from "../ui/Modal";
import { PLURAL_FORMS, pluralize } from "../../lib/pluralize";
import { toast } from "../../lib/toast";

type Props = {
  open: boolean;
  onClose: () => void;
  userId: string | null;
  username: string | null;
  datasetsCount: number;
  analysesCount: number;
  reportsCount: number;
  /** Дёргается после успешного DELETE — например, чтобы парент закрыл
   *  свою AdminUserDetailModal. */
  onDeleted?: () => void;
};

export function DeleteUserConfirmModal({
  open,
  onClose,
  userId,
  username,
  datasetsCount,
  analysesCount,
  reportsCount,
  onDeleted,
}: Props) {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: (id: string) => adminApi.removeUser(id),
    onSuccess: (_data, id) => {
      // Кэш detail удалённого юзера не нужен.
      queryClient.removeQueries({ queryKey: ["admin", "user", id] });
      // Инвалидируем все страницы листинга и сводку — счётчики системы
      // сдвинутся вместе с удалением каскада.
      queryClient.invalidateQueries({
        predicate: (q) =>
          q.queryKey[0] === "admin" &&
          (q.queryKey[1] === "users" || q.queryKey[1] === "stats"),
      });
      toast.success(`Пользователь ${username ?? "—"} удалён`);
      deleteMutation.reset();
      onClose();
      onDeleted?.();
    },
    onError: (err) => {
      const detail = extractDetail(err);
      if (detail && detail.includes("your own admin account")) {
        toast.warning("Нельзя удалить собственный аккаунт администратора");
        deleteMutation.reset();
        onClose();
        return;
      }
      if (detail && detail.includes("last admin account")) {
        toast.warning("Нельзя удалить последнего администратора в системе");
        deleteMutation.reset();
        onClose();
        return;
      }
      // Прочие ошибки — оставляем модалку, рендерим inline-блок.
    },
  });

  const isDeleting = deleteMutation.isPending;
  const isInlineError =
    deleteMutation.isError && !isHandledStatusError(deleteMutation.error);
  const errorText = isInlineError
    ? "Не удалось удалить. Повторите попытку или обратитесь к администратору."
    : null;

  const title = username
    ? `Удалить пользователя ${username}?`
    : "Удалить пользователя?";

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
      <Body
        datasetsCount={datasetsCount}
        analysesCount={analysesCount}
        reportsCount={reportsCount}
      />

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
          onClick={() => userId && deleteMutation.mutate(userId)}
          disabled={isDeleting || userId === null}
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

function Body({
  datasetsCount,
  analysesCount,
  reportsCount,
}: {
  datasetsCount: number;
  analysesCount: number;
  reportsCount: number;
}) {
  if (datasetsCount === 0 && analysesCount === 0 && reportsCount === 0) {
    return (
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        У пользователя нет датасетов, анализов и отчётов. Действие необратимо.
      </p>
    );
  }
  const dsetsWord = pluralize(datasetsCount, PLURAL_FORMS.dataset);
  const anWord = pluralize(analysesCount, PLURAL_FORMS.analysis);
  const repWord = pluralize(reportsCount, PLURAL_FORMS.pdfReport);
  return (
    <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
      Будут удалены:{" "}
      <span className="font-mono text-[0.875rem]">{datasetsCount}</span>{" "}
      {dsetsWord},{" "}
      <span className="font-mono text-[0.875rem]">{analysesCount}</span>{" "}
      {anWord},{" "}
      <span className="font-mono text-[0.875rem]">{reportsCount}</span>{" "}
      {repWord}. Действие необратимо.
    </p>
  );
}

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

function isHandledStatusError(err: unknown): boolean {
  const detail = extractDetail(err);
  if (!detail) return false;
  return (
    detail.includes("your own admin account") ||
    detail.includes("last admin account")
  );
}
