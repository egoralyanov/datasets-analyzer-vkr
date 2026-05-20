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
// Стиль (Sprint 5, Phase 3.7): scientific/archive — §-заголовок,
// нумерованный каталог анализов, статус как тонкая текстовая отметка
// (без bg-fill), пагинация в archive-стиле.
//
// Фронт-тестов нет (Vitest/Testing Library не подключены).
import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Trash2 } from "lucide-react";
import { analysesApi } from "../api/analyses";
import { DeleteAnalysisModal } from "../components/analysis/DeleteAnalysisModal";
import { computeDuration } from "../lib/duration";
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

// Семантический цвет статуса (только цвет текста + 1px-обводка), без
// заливки. По принципу п.7 манифеста DESIGN_TOKENS.md.
const STATUS_TEXT: Record<AnalysisStatus, string> = {
  pending: "text-paper-500 border-paper-400",
  running: "text-info-700 border-info-500",
  done: "text-success-700 border-success-500",
  failed: "text-critical-700 border-critical-500",
};

const FILTER_OPTIONS: { value: AnalysisStatus | "all"; label: string }[] = [
  { value: "all", label: "ВСЕ" },
  { value: "done", label: "ГОТОВЫЕ" },
  { value: "running", label: "ВЫПОЛНЯЮТСЯ" },
  { value: "failed", label: "С ОШИБКОЙ" },
];

export function History() {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<AnalysisStatus | "all">(
    "all",
  );
  // Какой анализ собираемся удалять (null = модалка закрыта). Храним
  // целиком элемент списка, чтобы пробросить filename / task / dates
  // в DeleteAnalysisModal без повторного запроса.
  const [toDelete, setToDelete] = useState<AnalysisListItem | null>(null);

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
    <div className="mx-auto max-w-[1100px] px-8 py-12 lg:px-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        ЖУРНАЛ ЗАПУСКОВ АНАЛИЗА
      </p>
      <h1 className="mt-2 flex items-baseline gap-3 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
        <span className="font-sans text-base font-medium text-paper-400">
          §
        </span>
        История анализов
      </h1>

      <div className="mt-8 flex flex-wrap items-baseline gap-x-6 gap-y-2 border-b border-paper-300 pb-3">
        <span className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
          ФИЛЬТР
        </span>
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          {FILTER_OPTIONS.map((opt) => {
            const active = statusFilter === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onChangeFilter(opt.value)}
                className={`border-b-2 pb-0.5 font-sans text-xs font-medium uppercase tracking-wider transition-colors ${
                  active
                    ? "border-ink-700 text-ink-900"
                    : "border-transparent text-paper-500 hover:text-ink-700"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-6">
        {query.isLoading && <SpinnerBox />}
        {query.isError && (
          <ErrorBox message="Не удалось загрузить список анализов." />
        )}
        {query.data && query.data.items.length === 0 && (
          <EmptyState hasFilter={statusFilter !== "all"} />
        )}
        {query.data && query.data.items.length > 0 && (
          <>
            <ol className="divide-y divide-paper-200 border-y border-paper-200">
              {query.data.items.map((item, idx) => (
                <AnalysisRow
                  key={item.id}
                  item={item}
                  index={(page - 1) * PAGE_SIZE + idx + 1}
                  onDelete={() => setToDelete(item)}
                />
              ))}
            </ol>
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

      <DeleteAnalysisModal
        open={toDelete !== null}
        onClose={() => setToDelete(null)}
        analysisId={toDelete?.id ?? null}
        datasetFilename={toDelete?.dataset_name ?? null}
        taskTypeCode={toDelete?.recommended_task_type ?? null}
        completedAt={toDelete?.finished_at ?? null}
        startedAt={toDelete?.started_at ?? new Date().toISOString()}
      />
    </div>
  );
}

export default History;

function AnalysisRow({
  item,
  index,
  onDelete,
}: {
  item: AnalysisListItem;
  index: number;
  onDelete: () => void;
}) {
  const startedAtFmt = formatDateTime(item.started_at);
  const durationFmt = computeDuration(item.started_at, item.finished_at);

  // Trash2 — отдельная кнопка-сиблинг для Link. Внутрь <a> кнопку класть
  // нельзя (невалидный HTML + Safari может проглотить клик). Hover-фон
  // строки поднят на <li role="group">, чтобы при наведении на иконку
  // фон не отскакивал, как было бы при hover'е на самом Link.
  return (
    <li className="group relative transition-colors hover:bg-paper-100/40">
      <Link
        to={`/analyses/${item.id}`}
        className="grid grid-cols-[2.5rem_1fr_auto] items-start gap-4 py-4 pl-1 pr-12"
      >
        <span className="pt-0.5 font-mono text-sm text-paper-400">
          {String(index).padStart(2, "0")}.
        </span>
        <div className="min-w-0">
          <p className="truncate font-serif text-[1.0625rem] font-semibold leading-snug text-paper-900">
            {item.dataset_name}
            {item.dataset_deleted && (
              <span className="ml-2 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
                (удалён)
              </span>
            )}
          </p>
          <p className="mt-1 font-mono text-xs text-paper-500">
            {startedAtFmt}
            {durationFmt && item.status === "done" && (
              <>
                <span className="mx-2">·</span>
                {durationFmt}
              </>
            )}
            <span className="mx-2">·</span>
            <span className="text-paper-400">{item.id.slice(0, 8)}</span>
          </p>
          {item.target_column && (
            <p className="mt-1 font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
              TARGET:{" "}
              <span className="font-mono normal-case tracking-normal text-paper-700">
                «{item.target_column}»
              </span>
            </p>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          <span
            className={`border px-2 py-0.5 font-sans text-[0.6875rem] font-medium uppercase tracking-wider ${
              STATUS_TEXT[item.status]
            }`}
          >
            {STATUS_LABELS[item.status]}
          </span>
          {item.recommended_task_type && (
            <span className="font-mono text-[0.6875rem] text-paper-600">
              {item.recommended_task_type.toLowerCase().replace(/_/g, " ")}
            </span>
          )}
        </div>
      </Link>
      <button
        type="button"
        aria-label="Удалить анализ"
        onClick={onDelete}
        className="absolute right-2 top-4 inline-flex h-8 w-8 items-center justify-center text-paper-500 transition-colors group-hover:text-critical-600 hover:!text-critical-700"
      >
        <Trash2 className="h-4 w-4" aria-hidden />
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

function EmptyState({ hasFilter }: { hasFilter: boolean }) {
  if (hasFilter) {
    return (
      <div className="border-l-[3px] border-paper-400 bg-paper-100/60 px-4 py-4 font-serif text-sm text-paper-600">
        По выбранному фильтру анализов нет.
      </div>
    );
  }
  return (
    <div className="border border-paper-300 bg-paper-50 px-8 py-12 text-center">
      <p className="font-serif text-[1rem] leading-relaxed text-paper-700">
        Анализов пока нет — загрузите датасет, чтобы запустить первый.
      </p>
      <Link
        to="/upload"
        className="mt-5 inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
      >
        ЗАГРУЗИТЬ ДАТАСЕТ
      </Link>
    </div>
  );
}

function SpinnerBox() {
  return (
    <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-8 font-sans text-sm text-paper-600">
      <Loader2 className="h-5 w-5 animate-spin text-ink-700" />
      <span>Загрузка истории…</span>
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

function formatDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return dateFormatter.format(d);
}

