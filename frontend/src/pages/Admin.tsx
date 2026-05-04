// Страница админ-панели: сводка по системе + пагинированный список
// пользователей. Доступ — через RequireAuth → RequireAdmin (см. App.tsx).
//
// Фронт-тесты не настроены (vitest/Testing Library не подключены) —
// gap из Phase 5 сохраняется. TODO: добавить при настройке инфраструктуры.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Database,
  FileSearch,
  FileText,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Users,
} from "lucide-react";
import { adminApi } from "../api/admin";
import type {
  AdminStats,
  AdminUserListItem,
  AdminUserListResponse,
} from "../types/admin";

const PAGE_SIZE = 20;

export function Admin() {
  const [page, setPage] = useState(1);

  const stats = useQuery<AdminStats>({
    queryKey: ["admin", "stats"],
    queryFn: () => adminApi.getStats(),
    staleTime: 0,
    refetchOnMount: "always",
  });

  const users = useQuery<AdminUserListResponse>({
    queryKey: ["admin", "users", page],
    queryFn: () => adminApi.listUsers({ page, size: PAGE_SIZE }),
    staleTime: 0,
    refetchOnMount: "always",
  });

  const onRefresh = () => {
    stats.refetch();
    users.refetch();
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-blue-600" />
        <h1 className="text-2xl font-semibold text-slate-900">Админ-панель</h1>
      </div>

      <div className="mt-1 flex items-center gap-3 text-xs text-slate-500">
        <span>{formatLastUpdated(stats.dataUpdatedAt)}</span>
        <button
          type="button"
          onClick={onRefresh}
          disabled={stats.isFetching || users.isFetching}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {stats.isFetching || users.isFetching ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
          Обновить
        </button>
      </div>

      <section className="mt-6">
        {stats.isLoading && <SpinnerBox label="Загрузка статистики…" />}
        {stats.isError && <ErrorBox message="Не удалось загрузить статистику." />}
        {stats.data && <StatsGrid data={stats.data} />}
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-slate-900">Пользователи</h2>
        <div className="mt-3">
          {users.isLoading && <SpinnerBox label="Загрузка пользователей…" />}
          {users.isError && (
            <ErrorBox message="Не удалось загрузить список пользователей." />
          )}
          {users.data && users.data.items.length > 0 && (
            <>
              <ul className="space-y-2">
                {users.data.items.map((u) => (
                  <li key={u.id}>
                    <UserRow item={u} />
                  </li>
                ))}
              </ul>
              <Pagination
                page={users.data.page}
                pages={users.data.pages}
                total={users.data.total}
                onPrev={() => setPage((p) => Math.max(1, p - 1))}
                onNext={() =>
                  setPage((p) =>
                    users.data && p < users.data.pages ? p + 1 : p,
                  )
                }
                isFetching={users.isFetching}
              />
            </>
          )}
          {users.data && users.data.items.length === 0 && (
            <div className="rounded-md border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
              Пользователей пока нет.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default Admin;

// =============================================================================
//                               БЛОКИ
// =============================================================================

function StatsGrid({ data }: { data: AdminStats }) {
  return (
    <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
      <StatCard
        icon={<Users className="h-4 w-4 text-blue-600" />}
        label="Пользователи"
        value={data.total_users}
      />
      <StatCard
        icon={<Database className="h-4 w-4 text-emerald-600" />}
        label="Датасеты"
        value={data.total_datasets}
      />
      <StatCard
        icon={<FileSearch className="h-4 w-4 text-violet-600" />}
        label="Анализы"
        value={data.total_analyses}
      />
      <StatCard
        icon={<FileText className="h-4 w-4 text-orange-600" />}
        label="PDF-отчёты"
        value={data.total_reports}
      />
      <RateCard
        label="Успешные анализы"
        rate={data.analyses_success_rate}
      />
      <RateCard
        label="Успешные отчёты"
        rate={data.reports_success_rate}
      />
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-900">
        {value.toLocaleString("ru-RU")}
      </div>
    </div>
  );
}

function RateCard({
  label,
  rate,
}: {
  label: string;
  rate: number | null;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold text-slate-900">
        {rate === null ? (
          <span className="text-slate-400">—</span>
        ) : (
          `${(rate * 100).toFixed(1)}%`
        )}
      </div>
    </div>
  );
}

function UserRow({ item }: { item: AdminUserListItem }) {
  const isAdmin = item.role === "admin";
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-900">{item.username}</p>
          <p className="mt-0.5 truncate text-sm text-slate-600">{item.email}</p>
          <p className="mt-1 text-xs text-slate-500">
            Зарегистрирован {formatDate(item.created_at)}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1.5">
          <span
            className={
              isAdmin
                ? "rounded-md border border-blue-200 bg-blue-100 px-2 py-0.5 text-xs text-blue-800"
                : "rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 text-xs text-slate-700"
            }
          >
            {isAdmin ? "admin" : "user"}
          </span>
          <span className="text-xs text-slate-500">
            датасетов:{" "}
            <span className="font-mono text-slate-700">
              {item.datasets_count}
            </span>{" "}
            · анализов:{" "}
            <span className="font-mono text-slate-700">
              {item.analyses_count}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}

function Pagination({
  page,
  pages,
  total,
  onPrev,
  onNext,
  isFetching,
}: {
  page: number;
  pages: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  isFetching: boolean;
}) {
  return (
    <div className="mt-5 flex items-center justify-between text-sm">
      <span className="text-slate-500">
        Всего: <span className="font-mono text-slate-700">{total}</span>
        {isFetching ? (
          <Loader2 className="ml-2 inline h-3.5 w-3.5 animate-spin text-slate-400" />
        ) : null}
      </span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onPrev}
          disabled={page <= 1}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-3 py-1 text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ChevronLeft className="h-4 w-4" />
          Предыдущая
        </button>
        <span className="text-slate-600">
          Стр. <span className="font-mono">{page}</span> из{" "}
          <span className="font-mono">{pages}</span>
        </span>
        <button
          type="button"
          onClick={onNext}
          disabled={page >= pages}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-3 py-1 text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Следующая
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function SpinnerBox({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 rounded-md border border-slate-200 bg-white p-6 text-slate-600">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>{label}</span>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      {message}
    </div>
  );
}

// =============================================================================
//                               ФОРМАТИРОВАНИЕ
// =============================================================================

const dateFormatter = new Intl.DateTimeFormat("ru-RU", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : dateFormatter.format(d);
}

function formatLastUpdated(timestamp: number): string {
  if (!timestamp) return "Данные ещё не загружены";
  const d = new Date(timestamp);
  return `Обновлено в ${d.toLocaleTimeString("ru-RU")}`;
}
