// Страница «История анализов» — пагинированный список с фильтром по статусу.
//
// Источник данных: GET /api/analyses?page=&size=&status= (см. бэк-эндпоинт
// list_my_analyses, контракт AnalysisListResponse). Сортировка
// started_at DESC; одна строка — кликабельный <Link>-блок (row-card),
// весь анализ открывается переходом на /analyses/{id}.
//
// React Query конфигурация: refetchOnMount: 'always' и staleTime: 0 — чтобы
// при возврате с /analyses/{id} (где могло поменяться состояние baseline /
// reports / status) список перерисовался от свежих данных, а не из stale
// кэша. Защита от мелкого UX-зуда «открыл анализ → сгенерировал отчёт →
// вернулся, а в списке всё по-старому».
//
// Фронт-тестов нет (Vitest/Testing Library не подключены) — TODO добавить
// при настройке тестовой инфраструктуры.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  History as HistoryIcon,
  Loader2,
  Upload as UploadIcon,
} from "lucide-react";
import { analysesApi } from "../api/analyses";
import type {
  AnalysisListItem,
  AnalysisListResponse,
  AnalysisStatus,
} from "../types/analysis";

const PAGE_SIZE = 20;

const STATUS_LABELS: Record<AnalysisStatus, string> = {
  pending: "В очереди",
  running: "Выполняется",
  done: "Готово",
  failed: "Ошибка",
};

const STATUS_BADGE_CLASSES: Record<AnalysisStatus, string> = {
  pending: "bg-slate-100 text-slate-700 border border-slate-200",
  running: "bg-blue-100 text-blue-800 border border-blue-200",
  done: "bg-emerald-100 text-emerald-800 border border-emerald-200",
  failed: "bg-red-100 text-red-800 border border-red-200",
};

const FILTER_OPTIONS: { value: AnalysisStatus | "all"; label: string }[] = [
  { value: "all", label: "Все" },
  { value: "done", label: "Готовые" },
  { value: "running", label: "Выполняются" },
  { value: "failed", label: "С ошибкой" },
];

export function History() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<AnalysisStatus | "all">("all");

  const query = useQuery<AnalysisListResponse>({
    queryKey: ["history", page, statusFilter],
    queryFn: () =>
      analysesApi.list({
        page,
        size: PAGE_SIZE,
        status: statusFilter === "all" ? null : statusFilter,
      }),
    staleTime: 0,
    refetchOnMount: "always",
  });

  const onChangeFilter = (value: AnalysisStatus | "all") => {
    setStatusFilter(value);
    setPage(1);
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <div className="flex items-center gap-2">
        <HistoryIcon className="h-5 w-5 text-blue-600" />
        <h1 className="text-2xl font-semibold text-slate-900">
          История анализов
        </h1>
      </div>

      <div className="mt-5 flex items-center gap-2 text-sm">
        <span className="text-slate-600">Фильтр по статусу:</span>
        <div className="flex flex-wrap gap-1.5">
          {FILTER_OPTIONS.map((opt) => {
            const active = statusFilter === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onChangeFilter(opt.value)}
                className={
                  active
                    ? "rounded-md bg-blue-600 px-3 py-1 text-white"
                    : "rounded-md border border-slate-200 bg-white px-3 py-1 text-slate-700 hover:bg-slate-50"
                }
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-5">
        {query.isLoading && <SpinnerBox />}
        {query.isError && (
          <ErrorBox message="Не удалось загрузить список анализов." />
        )}
        {query.data && query.data.items.length === 0 && (
          <EmptyState hasFilter={statusFilter !== "all"} />
        )}
        {query.data && query.data.items.length > 0 && (
          <>
            <ul className="space-y-2">
              {query.data.items.map((item) => (
                <li key={item.id}>
                  <AnalysisRow item={item} />
                </li>
              ))}
            </ul>
            <Pagination
              page={query.data.page}
              pages={query.data.pages}
              total={query.data.total}
              onPrev={() => setPage((p) => Math.max(1, p - 1))}
              onNext={() =>
                setPage((p) =>
                  query.data && p < query.data.pages ? p + 1 : p,
                )
              }
              isFetching={query.isFetching}
            />
          </>
        )}
      </div>
    </div>
  );
}

export default History;

// =============================================================================
//                               СОСТОЯНИЯ И ЭЛЕМЕНТЫ
// =============================================================================

function AnalysisRow({ item }: { item: AnalysisListItem }) {
  const startedAtFmt = formatDateTime(item.started_at);
  const durationFmt = computeDuration(item.started_at, item.finished_at);

  return (
    <Link
      to={`/analyses/${item.id}`}
      className="block rounded-md border border-slate-200 bg-white p-4 transition hover:border-blue-300 hover:bg-slate-50"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-900">
            {item.dataset_name}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Запущен {startedAtFmt}
            {durationFmt && item.status === "done" ? (
              <span className="ml-2 text-slate-400">
                (выполнялся: {durationFmt})
              </span>
            ) : null}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {item.recommended_task_type ? (
            <span
              className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs text-slate-600"
              title="Рекомендованный тип задачи"
            >
              {item.recommended_task_type}
            </span>
          ) : null}
          <span
            className={`rounded-md px-2 py-0.5 text-xs ${STATUS_BADGE_CLASSES[item.status]}`}
          >
            {STATUS_LABELS[item.status]}
          </span>
        </div>
      </div>

      {item.target_column ? (
        <p className="mt-2 text-xs text-slate-500">
          Целевая переменная:{" "}
          <span className="font-mono text-slate-700">{item.target_column}</span>
        </p>
      ) : null}
    </Link>
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

function EmptyState({ hasFilter }: { hasFilter: boolean }) {
  if (hasFilter) {
    return (
      <div className="rounded-md border border-slate-200 bg-white p-8 text-center text-sm text-slate-600">
        По выбранному фильтру анализов нет.
      </div>
    );
  }
  return (
    <div className="rounded-md border border-slate-200 bg-white p-10 text-center">
      <p className="text-slate-700">
        Анализов пока нет — загрузите датасет на странице загрузки.
      </p>
      <Link
        to="/upload"
        className="mt-4 inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
      >
        <UploadIcon className="h-4 w-4" />
        Загрузить датасет
      </Link>
    </div>
  );
}

function SpinnerBox() {
  return (
    <div className="flex items-center justify-center gap-3 rounded-md border border-slate-200 bg-white p-8 text-slate-600">
      <Loader2 className="h-5 w-5 animate-spin" />
      <span>Загрузка истории…</span>
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

function formatDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return dateFormatter.format(d);
}

function computeDuration(
  startedIso: string,
  finishedIso: string | null,
): string | null {
  if (!finishedIso) return null;
  const startMs = Date.parse(startedIso);
  const endMs = Date.parse(finishedIso);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null;
  const seconds = Math.max(0, Math.round((endMs - startMs) / 1000));
  if (seconds < 60) return `${seconds} сек`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest === 0 ? `${minutes} мин` : `${minutes} мин ${rest} сек`;
}
