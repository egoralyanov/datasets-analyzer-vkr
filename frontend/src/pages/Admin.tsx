// Страница админ-панели: сводка по системе + пагинированный список
// пользователей. Доступ — через RequireAuth → RequireAdmin (см. App.tsx).
//
// Стиль (Sprint 5, Phase 3.10): scientific/archive — §-заголовок, плоские
// stat-блоки без скруглений и теней (цифры в Plex Serif, лейблы uppercase
// tracking), пользователи как нумерованный каталог с тонкими бейджами роли.
//
// Фронт-тесты не настроены (vitest/Testing Library не подключены).
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { adminApi } from "../api/admin";
import { AdminUserDetailModal } from "../components/admin/AdminUserDetailModal";
import { RolePill } from "../components/admin/RolePill";
import type {
  AdminStats,
  AdminUserListItem,
  AdminUserListResponse,
} from "../types/admin";

const PAGE_SIZE = 20;

export function Admin() {
  const [page, setPage] = useState(1);
  // Sprint 6, Phase 5.4: клик по строке открывает AdminUserDetailModal с
  // деталями выбранного юзера. null — модалка закрыта.
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

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

  const isFetching = stats.isFetching || users.isFetching;

  return (
    <div className="mx-auto max-w-[1100px] px-8 py-12 lg:px-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        АДМИНИСТРАТИВНЫЙ КОНТУР · ТРЕБУЕТ РОЛЬ ADMIN
      </p>
      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-4">
        <h1 className="flex items-baseline gap-3 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
          <span className="font-sans text-base font-medium text-paper-400">
            §
          </span>
          Админ-панель
        </h1>
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-xs text-paper-500">
            {formatLastUpdated(stats.dataUpdatedAt)}
          </span>
          <button
            type="button"
            onClick={onRefresh}
            disabled={isFetching}
            className="inline-flex items-center gap-1.5 border border-ink-700 bg-paper-50 px-3 py-1.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isFetching ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : null}
            ОБНОВИТЬ
          </button>
        </div>
      </div>

      <section className="mt-8">
        <SubsectionHeader title="Сводка по системе" />
        {stats.isLoading && <SpinnerBox label="Загрузка статистики…" />}
        {stats.isError && (
          <ErrorBox message="Не удалось загрузить статистику." />
        )}
        {stats.data && <StatsGrid data={stats.data} />}
      </section>

      <section className="mt-12">
        <SubsectionHeader
          title="Пользователи"
          note={
            users.data
              ? `${users.data.total} ${pluralizeRu(
                  users.data.total,
                  "запись",
                  "записи",
                  "записей",
                )}`
              : undefined
          }
        />
        {users.isLoading && <SpinnerBox label="Загрузка пользователей…" />}
        {users.isError && (
          <ErrorBox message="Не удалось загрузить список пользователей." />
        )}
        {users.data && users.data.items.length > 0 && (
          <>
            <ol className="divide-y divide-paper-200 border-y border-paper-200">
              {users.data.items.map((u, idx) => (
                <UserRow
                  key={u.id}
                  item={u}
                  index={(page - 1) * PAGE_SIZE + idx + 1}
                  onOpen={() => setSelectedUserId(u.id)}
                />
              ))}
            </ol>
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
          <div className="border-l-[3px] border-paper-400 bg-paper-100/60 px-4 py-3 font-serif text-sm text-paper-600">
            Пользователей пока нет.
          </div>
        )}
      </section>

      <AdminUserDetailModal
        open={selectedUserId !== null}
        onClose={() => setSelectedUserId(null)}
        userId={selectedUserId}
      />
    </div>
  );
}

export default Admin;

function SubsectionHeader({
  title,
  note,
}: {
  title: string;
  note?: string;
}) {
  return (
    <header className="mb-5 flex items-baseline justify-between gap-4 border-b border-paper-300 pb-2">
      <h2 className="font-serif text-[1.5rem] font-semibold leading-snug tracking-tight text-paper-900">
        {title}
      </h2>
      {note && (
        <span className="font-sans text-xs uppercase tracking-wider text-paper-500">
          {note}
        </span>
      )}
    </header>
  );
}

function StatsGrid({ data }: { data: AdminStats }) {
  // 4 stat-cards в первой строке + 2 rate-cards во второй (каждая span=2 на
  // широких экранах). На узких — 2 колонки, всё ложится без пустых ячеек.
  // Иначе grid-bg paper-300 просвечивал бы как пустой серый разделитель.
  return (
    <div className="grid grid-cols-2 gap-x-px gap-y-px border border-paper-300 bg-paper-300 lg:grid-cols-4">
      <StatCard label="Пользователи" value={data.total_users} />
      <StatCard label="Датасеты" value={data.total_datasets} />
      <StatCard label="Анализы" value={data.total_analyses} />
      <StatCard label="PDF-отчёты" value={data.total_reports} />
      <RateCard
        label="Успешные анализы"
        rate={data.analyses_success_rate}
        wide
      />
      <RateCard
        label="Успешные отчёты"
        rate={data.reports_success_rate}
        wide
      />
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-paper-50 px-5 py-5">
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <p className="mt-2 font-serif text-[2.25rem] font-bold leading-none tracking-tight text-paper-900">
        {value.toLocaleString("ru-RU")}
      </p>
    </div>
  );
}

function RateCard({
  label,
  rate,
  wide = false,
}: {
  label: string;
  rate: number | null;
  wide?: boolean;
}) {
  const tone =
    rate === null
      ? "text-paper-400"
      : rate >= 0.95
        ? "text-success-700"
        : rate >= 0.7
          ? "text-warning-700"
          : "text-critical-700";
  return (
    <div
      className={`bg-paper-50 px-5 py-5 ${wide ? "lg:col-span-2" : ""}`}
    >
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <p
        className={`mt-2 font-mono text-[2rem] font-medium leading-none ${tone}`}
      >
        {rate === null ? "—" : `${(rate * 100).toFixed(1)}%`}
      </p>
    </div>
  );
}

function UserRow({
  item,
  index,
  onOpen,
}: {
  item: AdminUserListItem;
  index: number;
  onOpen: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="grid w-full grid-cols-[2.5rem_1fr_auto] items-start gap-4 px-1 py-4 text-left transition-colors hover:bg-paper-100/50"
      >
        <span className="pt-0.5 font-mono text-sm text-paper-400">
          {String(index).padStart(3, "0")}.
        </span>
        <div className="min-w-0">
          <p className="truncate font-serif text-[1.0625rem] font-semibold leading-snug text-paper-900">
            {item.username}
          </p>
          <p className="mt-1 truncate font-mono text-xs text-paper-700">
            {item.email}
          </p>
          <p className="mt-1 font-mono text-xs text-paper-500">
            Зарегистрирован {formatDate(item.created_at)}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          <RolePill role={item.role} />
          <span className="font-mono text-[0.6875rem] text-paper-500">
            dsets{" "}
            <span className="text-paper-700">{item.datasets_count}</span>
            <span className="mx-1">·</span>
            analyses{" "}
            <span className="text-paper-700">{item.analyses_count}</span>
          </span>
        </div>
      </button>
    </li>
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
    <div className="mt-6 flex flex-wrap items-baseline justify-between gap-3 border-t border-paper-300 pt-3">
      <span className="font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
        ВСЕГО{" "}
        <span className="ml-1 font-mono normal-case tracking-normal text-paper-700">
          {total}
        </span>
        {isFetching && (
          <Loader2 className="ml-2 inline h-3 w-3 animate-spin text-info-500" />
        )}
      </span>
      <div className="flex items-baseline gap-3">
        <button
          type="button"
          onClick={onPrev}
          disabled={page <= 1}
          className="border-b-2 border-transparent pb-0.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          ← ПРЕДЫДУЩАЯ
        </button>
        <span className="font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
          СТР.{" "}
          <span className="font-mono normal-case tracking-normal text-paper-700">
            {page}
          </span>{" "}
          ИЗ{" "}
          <span className="font-mono normal-case tracking-normal text-paper-700">
            {pages}
          </span>
        </span>
        <button
          type="button"
          onClick={onNext}
          disabled={page >= pages}
          className="border-b-2 border-transparent pb-0.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          СЛЕДУЮЩАЯ →
        </button>
      </div>
    </div>
  );
}

function SpinnerBox({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-6 font-sans text-sm text-paper-600">
      <Loader2 className="h-4 w-4 animate-spin text-ink-700" />
      <span>{label}</span>
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
  return `обновлено ${d.toLocaleTimeString("ru-RU")}`;
}

function pluralizeRu(
  n: number,
  one: string,
  few: string,
  many: string,
): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}
