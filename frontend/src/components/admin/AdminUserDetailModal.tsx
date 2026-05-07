// Детальная карточка пользователя в админ-панели (Sprint 6, Phase 5.4).
//
// Открывается из листинга /admin по клику на строку. Тянет
// GET /api/admin/users/{id} (см. backend/app/api/admin.py, get_user_detail)
// в queryKey ['admin', 'user', userId]. Пока запрос идёт — на месте сетки
// данных рендерится спиннер; заголовок и крестик активны.
//
// Внутри есть кнопка «УДАЛИТЬ ПОЛЬЗОВАТЕЛЯ», которая открывает
// DeleteUserConfirmModal поверх. На время, пока вторая модалка открыта,
// сюда передаётся inactive=true (см. Modal.tsx) — клавиатура и фокус
// принадлежат верхней. После успешного удаления вторая модалка дёргает
// onDeleted → закрывает эту тоже через onClose.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Trash2 } from "lucide-react";

import { adminApi } from "../../api/admin";
import { Modal } from "../ui/Modal";
import { formatDateTime } from "../../lib/format";
import { DeleteUserConfirmModal } from "./DeleteUserConfirmModal";
import { RolePill } from "./RolePill";

type Props = {
  open: boolean;
  onClose: () => void;
  userId: string | null;
  /** Парент может среагировать на удаление — например, очистить
   *  selectedUserId. Дёргается ПОСЛЕ закрытия обеих модалок. */
  onDeleted?: () => void;
};

export function AdminUserDetailModal({
  open,
  onClose,
  userId,
  onDeleted,
}: Props) {
  const [confirmOpen, setConfirmOpen] = useState(false);

  const userQuery = useQuery({
    queryKey: ["admin", "user", userId],
    queryFn: () => adminApi.getUser(userId!),
    enabled: open && userId !== null,
    // Каждое открытие модалки берём свежие счётчики артефактов — между
    // открытиями admin мог дёрнуть delete-flow для другого юзера и
    // изменить общую сводку.
    staleTime: 0,
    refetchOnMount: "always",
  });

  const detail = userQuery.data;
  const title = detail?.username ?? "Загрузка…";

  return (
    <>
      <Modal
        open={open}
        onClose={() => {
          if (confirmOpen) return;
          onClose();
        }}
        title={title}
        subtitle="ПОЛЬЗОВАТЕЛЬ"
        size="lg"
        inactive={confirmOpen}
      >
        {userQuery.isLoading && <Spinner />}
        {userQuery.isError && (
          <ErrorBox message="Не удалось загрузить данные пользователя." />
        )}
        {detail && (
          <>
            <DataGrid detail={detail} />

            <div className="my-6 border-t border-paper-300" />

            <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
              ДЕЙСТВИЯ
            </p>
            <div className="mt-3">
              <button
                type="button"
                onClick={() => setConfirmOpen(true)}
                className="inline-flex items-center gap-2 border border-critical-500 bg-critical-500 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:border-critical-600 hover:bg-critical-600"
              >
                <Trash2 className="h-3.5 w-3.5" />
                УДАЛИТЬ ПОЛЬЗОВАТЕЛЯ
              </button>
              <p className="mt-2 font-sans text-xs leading-relaxed text-paper-500">
                Удалит все датасеты, анализы и PDF-отчёты пользователя.
                Действие необратимо.
              </p>
            </div>
          </>
        )}
      </Modal>

      <DeleteUserConfirmModal
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        userId={userId}
        username={detail?.username ?? null}
        datasetsCount={detail?.datasets_count ?? 0}
        analysesCount={detail?.analyses_count ?? 0}
        reportsCount={detail?.reports_count ?? 0}
        onDeleted={() => {
          // Вторая модалка уже закрылась через свой onClose. Закрываем
          // и эту, потом сообщаем паренту через onDeleted.
          onClose();
          onDeleted?.();
        }}
      />
    </>
  );
}

function DataGrid({
  detail,
}: {
  detail: {
    email: string;
    username: string;
    role: "user" | "admin";
    created_at: string;
    datasets_count: number;
    analyses_count: number;
    reports_count: number;
  };
}) {
  return (
    <div className="grid grid-cols-1 gap-x-6 gap-y-4 lg:grid-cols-2">
      <Field label="EMAIL" value={detail.email} mono />
      <Field label="USERNAME" value={detail.username} mono />
      <Field
        label="РОЛЬ"
        value={<RolePill role={detail.role} />}
      />
      <Field
        label="ЗАРЕГИСТРИРОВАН"
        value={formatDateTime(detail.created_at)}
        mono
      />
      <StatField label="ДАТАСЕТЫ" value={detail.datasets_count} />
      <StatField label="АНАЛИЗЫ" value={detail.analyses_count} />
      <StatField label="PDF-ОТЧЁТЫ" value={detail.reports_count} />
    </div>
  );
}

function Field({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div>
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </p>
      <div
        className={`mt-1 ${mono ? "font-mono text-sm" : "font-sans text-sm"} text-paper-800`}
      >
        {value}
      </div>
    </div>
  );
}

function StatField({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-2xl font-medium leading-none text-paper-900">
        {value.toLocaleString("ru-RU")}
      </p>
    </div>
  );
}

function Spinner() {
  return (
    <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-6 font-sans text-sm text-paper-600">
      <Loader2 className="h-4 w-4 animate-spin text-ink-700" />
      <span>Загрузка пользователя…</span>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
      {message}
    </div>
  );
}
