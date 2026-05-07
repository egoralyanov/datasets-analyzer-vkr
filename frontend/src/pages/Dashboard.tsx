// Dashboard для авторизованного пользователя (Sprint 6, Phase 7).
//
// Заменяет inline-AuthenticatedDashboard, живший в Landing.tsx до Phase 7.
// Landing.tsx остаётся диспатчером (guest vs auth), а Dashboard вынесен
// сюда как полноценная страница: hero → mini-stats → две колонки
// (последние анализы + последние датасеты) → финальный CTA. Эстетика
// scientific/archive с чередующимися band-полосами paper-50/100, как
// гостевая главная Phase 6.
//
// Источники данных:
// - GET /api/me/stats (Phase 7 backend) — четыре счётчика для mini-stats.
// - GET /api/analyses?page=1&size=5 — последние 5 анализов.
// - GET /api/datasets — полный список без пагинации; на фронте slice(0, 5).
//   Расширение API ради косметики dashboard сейчас оверинжиниринг (Phase 7
//   ответ Архитектора, вариант A); зона роста зафиксирована в РПЗ.
//
// Клик по строке датасета ведёт на /upload?open={id} — Upload.tsx читает
// query-параметр и автоматически дёргает openMutation, открывая превью
// ровно того файла, на который кликнули. Минимально инвазивное решение
// (один useEffect в Upload.tsx) вместо тяжёлого inline-preview на dashboard.
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { FileText, Loader2 } from "lucide-react";
import { analysesApi } from "../api/analyses";
import { datasetsApi } from "../api/datasets";
import { meApi } from "../api/me";
import { StatCell } from "../components/ui/StatCell";
import { formatBytes, formatDateTime, formatNumber } from "../lib/format";
import { TASK_TYPE_LABEL } from "../lib/taskTypes";
import { useAuthStore } from "../store/authStore";
import type {
  AnalysisListItem,
  AnalysisStatus,
  TaskTypeCode,
} from "../types/analysis";
import type { Dataset } from "../types/dataset";
import type { UserStats } from "../types/user";

const ANALYSES_PAGE_SIZE = 5;
const DATASETS_PREVIEW_LIMIT = 5;

export function Dashboard() {
  const user = useAuthStore((s) => s.user)!;

  const statsQuery = useQuery<UserStats>({
    queryKey: ["me", "stats"],
    queryFn: () => meApi.getStats(),
    // Каждое возвращение на dashboard — свежие счётчики. Юзер мог за это
    // время сгенерировать отчёт или удалить анализ.
    staleTime: 0,
    refetchOnMount: "always",
  });

  const analysesQuery = useQuery({
    queryKey: ["dashboard", "analyses-recent"],
    queryFn: () => analysesApi.list({ page: 1, size: ANALYSES_PAGE_SIZE }),
    staleTime: 0,
    refetchOnMount: "always",
  });

  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: () => datasetsApi.list(),
  });

  const stats = statsQuery.data;
  const allZero =
    stats !== undefined &&
    stats.datasets_count === 0 &&
    stats.analyses_count === 0 &&
    stats.successful_analyses_count === 0 &&
    stats.reports_count === 0;
  const hasDatasets = (stats?.datasets_count ?? 0) > 0;

  const recentAnalyses = analysesQuery.data?.items ?? [];
  const recentDatasets = (datasetsQuery.data ?? []).slice(
    0,
    DATASETS_PREVIEW_LIMIT,
  );

  return (
    <div className="bg-paper-50">
      <Hero username={user.username} hasDatasets={hasDatasets} />
      {stats && !allZero && <MiniStatsBand stats={stats} />}
      <RecentBand
        analyses={recentAnalyses}
        analysesLoading={analysesQuery.isLoading}
        datasets={recentDatasets}
        datasetsLoading={datasetsQuery.isLoading}
      />
      <FinalCtaBand hasDatasets={hasDatasets} />
    </div>
  );
}

export default Dashboard;

// ───────────────────────────────────────────────────────────────────────
// Hero
// ───────────────────────────────────────────────────────────────────────
function Hero({
  username,
  hasDatasets,
}: {
  username: string;
  hasDatasets: boolean;
}) {
  return (
    <section className="bg-paper-50">
      <div className="mx-auto max-w-[1200px] px-8 py-16 lg:px-16 lg:py-20">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-600">
          АНАЛИЗАТОР · ПАНЕЛЬ
        </p>
        <h1 className="mt-3 font-serif text-[2.75rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[3.25rem] lg:text-[3.75rem]">
          С возвращением, {username}
        </h1>
        <p className="mt-4 max-w-prose font-serif text-[1rem] leading-relaxed text-paper-700 lg:text-[1.0625rem]">
          {hasDatasets
            ? "Загрузите новый датасет или продолжите работу с существующими."
            : "Загрузите первый датасет, чтобы получить рекомендации по типу задачи, проверки качества данных и базовую модель."}
        </p>
        <div className="mt-7">
          <Link
            to="/upload"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-6 py-3 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            ЗАГРУЗИТЬ ДАТАСЕТ
          </Link>
        </div>
      </div>
    </section>
  );
}

// ───────────────────────────────────────────────────────────────────────
// Mini-stats band
// ───────────────────────────────────────────────────────────────────────
function MiniStatsBand({ stats }: { stats: UserStats }) {
  const successHint =
    stats.analyses_count > 0
      ? `из ${stats.analyses_count}`
      : undefined;
  return (
    <section className="bg-paper-100">
      <div className="mx-auto max-w-[1200px] px-8 py-12 lg:px-16 lg:py-14">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-600">
          ВАША СТАТИСТИКА
        </p>
        <div className="mt-5 grid grid-cols-2 gap-px border border-paper-300 bg-paper-300 lg:grid-cols-4">
          <StatCell label="Датасеты" value={stats.datasets_count} />
          <StatCell label="Анализы" value={stats.analyses_count} />
          <StatCell
            label="Успешные"
            value={stats.successful_analyses_count}
            hint={successHint}
          />
          <StatCell label="PDF-отчёты" value={stats.reports_count} />
        </div>
      </div>
    </section>
  );
}

// ───────────────────────────────────────────────────────────────────────
// Recent (analyses + datasets)
// ───────────────────────────────────────────────────────────────────────
function RecentBand({
  analyses,
  analysesLoading,
  datasets,
  datasetsLoading,
}: {
  analyses: AnalysisListItem[];
  analysesLoading: boolean;
  datasets: Dataset[];
  datasetsLoading: boolean;
}) {
  return (
    <section className="bg-paper-50">
      <div className="mx-auto grid max-w-[1200px] gap-12 px-8 py-12 lg:grid-cols-2 lg:px-16 lg:py-16">
        <div>
          <SectionHeading
            title="Последние анализы"
            allHref="/history"
            allLabel="ВСЕ АНАЛИЗЫ →"
          />
          {analysesLoading ? (
            <Spinner label="Загрузка анализов…" />
          ) : analyses.length === 0 ? (
            <EmptyState
              title="Анализов пока нет"
              hint="Загрузите датасет — система автоматически выполнит профайлинг и предложит подходящий тип ML-задачи."
            />
          ) : (
            <ol className="divide-y divide-paper-200 border-y border-paper-200">
              {analyses.map((a) => (
                <RecentAnalysisRow key={a.id} item={a} />
              ))}
            </ol>
          )}
        </div>
        <div>
          <SectionHeading
            title="Последние датасеты"
            allHref="/upload"
            allLabel="ВСЕ ДАТАСЕТЫ →"
          />
          {datasetsLoading ? (
            <Spinner label="Загрузка датасетов…" />
          ) : datasets.length === 0 ? (
            <EmptyState
              title="Датасетов пока нет"
              hint="Перетащите CSV или XLSX на странице загрузки — анализ запустится автоматически."
            />
          ) : (
            <ol className="divide-y divide-paper-200 border-y border-paper-200">
              {datasets.map((d) => (
                <RecentDatasetRow key={d.id} item={d} />
              ))}
            </ol>
          )}
        </div>
      </div>
    </section>
  );
}

function SectionHeading({
  title,
  allHref,
  allLabel,
}: {
  title: string;
  allHref: string;
  allLabel: string;
}) {
  return (
    <header className="mb-4 flex items-baseline justify-between gap-4 border-b border-paper-300 pb-2">
      <h2 className="font-serif text-[1.5rem] font-semibold leading-snug tracking-tight text-paper-900">
        {title}
      </h2>
      <Link
        to={allHref}
        className="font-sans text-xs font-medium uppercase tracking-wider text-paper-600 underline-offset-2 hover:text-ink-700 hover:underline"
      >
        {allLabel}
      </Link>
    </header>
  );
}

const STATUS_LABELS: Record<AnalysisStatus, string> = {
  pending: "В очереди",
  running: "Выполняется",
  done: "Готово",
  failed: "Ошибка",
};

const STATUS_TONE: Record<AnalysisStatus, string> = {
  pending: "text-paper-500 border-paper-400",
  running: "text-info-700 border-info-500",
  done: "text-success-700 border-success-500",
  failed: "text-critical-700 border-critical-500",
};

function RecentAnalysisRow({ item }: { item: AnalysisListItem }) {
  const startedAt = formatDateTime(item.started_at);
  const taskLabel = item.recommended_task_type
    ? TASK_TYPE_LABEL[item.recommended_task_type as TaskTypeCode] ?? null
    : null;
  return (
    <li>
      <Link
        to={`/analyses/${item.id}`}
        className="grid grid-cols-[1fr_auto] items-start gap-4 px-1 py-3 transition-colors hover:bg-paper-100/40"
      >
        <div className="min-w-0">
          <p className="truncate font-serif text-[1rem] font-semibold leading-snug text-paper-900">
            {item.dataset_name}
          </p>
          <p className="mt-1 font-mono text-xs text-paper-500">{startedAt}</p>
          {taskLabel && (
            <p className="mt-1 font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
              {taskLabel}
            </p>
          )}
        </div>
        <span
          className={`shrink-0 border px-2 py-0.5 font-sans text-[0.6875rem] font-medium uppercase tracking-wider ${
            STATUS_TONE[item.status]
          }`}
        >
          {STATUS_LABELS[item.status]}
        </span>
      </Link>
    </li>
  );
}

function RecentDatasetRow({ item }: { item: Dataset }) {
  const sizeRows =
    item.n_rows !== null && item.n_cols !== null
      ? `${formatNumber(item.n_rows)} × ${formatNumber(item.n_cols)}`
      : null;
  return (
    <li>
      <Link
        to={`/upload?open=${item.id}`}
        className="block px-1 py-3 transition-colors hover:bg-paper-100/40"
      >
        <p className="truncate font-serif text-[0.9375rem] font-semibold leading-snug text-paper-900">
          {item.original_filename}
        </p>
        <p className="mt-1 font-mono text-xs text-paper-500">
          {sizeRows && <>{sizeRows} · </>}
          {formatBytes(item.file_size_bytes)} · {item.format.toUpperCase()}
        </p>
      </Link>
    </li>
  );
}

// ───────────────────────────────────────────────────────────────────────
// Final CTA
// ───────────────────────────────────────────────────────────────────────
function FinalCtaBand({ hasDatasets }: { hasDatasets: boolean }) {
  return (
    <section className="bg-paper-100">
      <div className="mx-auto flex max-w-[760px] flex-col items-center px-8 py-16 text-center lg:py-20">
        <h2 className="font-serif text-[1.75rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[2.25rem]">
          {hasDatasets ? "Анализ ещё одного датасета?" : "Готовы начать?"}
        </h2>
        <p className="mt-4 max-w-prose font-serif text-[1rem] leading-relaxed text-paper-700">
          Загрузите CSV или XLSX. Анализ выполняется в фоне за 5–15 секунд.
        </p>
        <div className="mt-7">
          <Link
            to="/upload"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-8 py-3 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            ЗАГРУЗИТЬ ДАТАСЕТ
          </Link>
        </div>
      </div>
    </section>
  );
}

// ───────────────────────────────────────────────────────────────────────
// Inline helpers
// ───────────────────────────────────────────────────────────────────────
function EmptyState({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="flex items-start gap-4 border border-paper-300 bg-paper-100/60 p-6">
      <FileText
        className="mt-1 h-6 w-6 shrink-0 text-paper-500"
        strokeWidth={1.5}
        aria-hidden
      />
      <div>
        <p className="font-serif text-[1.0625rem] font-semibold leading-snug text-paper-900">
          {title}
        </p>
        <p className="mt-1 font-sans text-sm leading-relaxed text-paper-600">
          {hint}
        </p>
      </div>
    </div>
  );
}

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-6 font-sans text-sm text-paper-600">
      <Loader2 className="h-4 w-4 animate-spin text-ink-700" />
      <span>{label}</span>
    </div>
  );
}

