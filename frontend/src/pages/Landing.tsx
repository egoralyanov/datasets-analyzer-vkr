// Главная страница.
//
// Стиль (Sprint 5, Phase 3.3): scientific/archive — серифный hero, под ним
// статичный «препарат» — уменьшённый кусок страницы анализа на эталонном
// датасете Iris, чтобы гость сразу увидел, что именно генерирует система.
// Никакого API, никакой интерактивности — данные захардкожены ниже.
//
// Авторизованный пользователь обслуживается отдельной веткой (см. Phase 3.4) —
// hero сворачивается, вместо демо появляется dashboard.
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../store/authStore";
import { datasetsApi } from "../api/datasets";
import { analysesApi } from "../api/analyses";
import { formatNumber } from "../lib/format";
import { TASK_TYPE_LABEL } from "../lib/taskTypes";
import type { TaskTypeCode } from "../types/analysis";

export function Landing() {
  const user = useAuthStore((s) => s.user);

  if (user) {
    return <AuthenticatedDashboard />;
  }

  return <GuestLanding />;
}

// Dashboard для авторизованного пользователя. Использует существующие
// эндпоинты: datasetsApi.list (полный список — берём длину) и
// analysesApi.list({page:1,size:1}) — даёт total для счётчика и items[0]
// как «последний анализ» (бэк сортирует started_at DESC).
//
// Onboarding-режим (datasets=0 и analyses=0) — большая CTA, без счётчиков.
function AuthenticatedDashboard() {
  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: () => datasetsApi.list(),
  });
  const analysesQuery = useQuery({
    queryKey: ["analyses-latest"],
    queryFn: () => analysesApi.list({ page: 1, size: 1 }),
  });

  const datasetsCount = datasetsQuery.data?.length ?? null;
  const analysesTotal = analysesQuery.data?.total ?? null;
  const latestAnalysis = analysesQuery.data?.items?.[0] ?? null;

  const isLoading = datasetsQuery.isLoading || analysesQuery.isLoading;
  const isOnboarding =
    !isLoading && datasetsCount === 0 && analysesTotal === 0;

  return (
    <div className="mx-auto max-w-[1100px] px-8 py-16 lg:px-16">
      {isOnboarding ? (
        <OnboardingHero />
      ) : (
        <DashboardView
          isLoading={isLoading}
          datasetsCount={datasetsCount}
          analysesTotal={analysesTotal}
          latestAnalysis={latestAnalysis}
        />
      )}
    </div>
  );
}

function OnboardingHero() {
  return (
    <div>
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        ДОБРО ПОЖАЛОВАТЬ В АНАЛИЗАТОР
      </p>
      <h1 className="mt-3 max-w-2xl font-serif text-[2.5rem] font-bold leading-tight tracking-tight text-paper-900">
        Загрузите первый датасет
      </h1>
      <p className="mt-4 max-w-2xl font-serif text-[1.0625rem] leading-relaxed text-paper-600">
        Поддерживаются файлы CSV и XLSX до 100 МБ. Кодировка и разделитель
        определяются автоматически. Анализ выполняется в фоне — вы получите
        профайл, флаги качества, рекомендацию типа задачи и базовую модель.
      </p>
      <div className="mt-8">
        <Link
          to="/upload"
          className="inline-flex items-center border border-ink-700 bg-ink-700 px-6 py-3 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
        >
          ЗАГРУЗИТЬ ПЕРВЫЙ ДАТАСЕТ
        </Link>
      </div>
    </div>
  );
}

function DashboardView({
  isLoading,
  datasetsCount,
  analysesTotal,
  latestAnalysis,
}: {
  isLoading: boolean;
  datasetsCount: number | null;
  analysesTotal: number | null;
  latestAnalysis: {
    id: string;
    dataset_name: string;
    started_at: string;
    recommended_task_type: string | null;
    status: string;
  } | null;
}) {
  return (
    <div>
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        АНАЛИЗАТОР · ПАНЕЛЬ
      </p>
      <h1 className="mt-3 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
        Рабочий стол
      </h1>

      <div className="mt-8">
        <Link
          to="/upload"
          className="inline-flex items-center border border-ink-700 bg-ink-700 px-6 py-3 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
        >
          ЗАГРУЗИТЬ ДАТАСЕТ
        </Link>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-2">
        <LatestAnalysisCard
          analysis={latestAnalysis}
          loading={isLoading}
        />
        <CounterCard
          label="Мои датасеты"
          count={datasetsCount}
          loading={isLoading}
          to="/upload"
          actionLabel="ПЕРЕЙТИ →"
          unitWord="датасет"
          unitWordPlural="датасета"
          unitWordPluralMany="датасетов"
        />
        <CounterCard
          label="История анализов"
          count={analysesTotal}
          loading={isLoading}
          to="/history"
          actionLabel="ОТКРЫТЬ →"
          unitWord="анализ"
          unitWordPlural="анализа"
          unitWordPluralMany="анализов"
        />
      </div>
    </div>
  );
}

function LatestAnalysisCard({
  analysis,
  loading,
}: {
  analysis: {
    id: string;
    dataset_name: string;
    started_at: string;
    recommended_task_type: string | null;
    status: string;
  } | null;
  loading: boolean;
}) {
  if (loading) {
    return <CardSkeleton label="Последний анализ" />;
  }
  if (!analysis) {
    // Сюда попадаем, если анализов нет, но есть датасеты — показываем
    // подсказку «запустите первый», вместо пустой плашки.
    return (
      <div className="border border-paper-300 bg-paper-50 p-6">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ПОСЛЕДНИЙ АНАЛИЗ
        </p>
        <p className="mt-3 font-serif text-[0.9375rem] leading-relaxed text-paper-600">
          Анализов пока нет. Откройте датасет и запустите первый — это
          занимает несколько секунд.
        </p>
        <Link
          to="/upload"
          className="mt-4 inline-block border-b border-ink-700 pb-0.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700"
        >
          К ДАТАСЕТАМ →
        </Link>
      </div>
    );
  }

  const startedAt = new Date(analysis.started_at).toLocaleString("ru-RU");
  const taskLabel = analysis.recommended_task_type
    ? TASK_TYPE_LABEL[analysis.recommended_task_type as TaskTypeCode] ??
      analysis.recommended_task_type
    : null;

  return (
    <Link
      to={`/analyses/${analysis.id}`}
      className="group border border-paper-300 bg-paper-50 p-6 transition-colors hover:border-ink-700"
    >
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        ПОСЛЕДНИЙ АНАЛИЗ
      </p>
      <h2 className="mt-2 font-serif text-[1.375rem] font-semibold leading-snug tracking-tight text-paper-900 group-hover:text-ink-900">
        {analysis.dataset_name}
      </h2>
      <dl className="mt-4 space-y-1 font-sans text-xs">
        <DashRow label="Запущен" value={startedAt} mono />
        <DashRow label="Статус" value={analysis.status.toUpperCase()} mono />
        {taskLabel && <DashRow label="Тип задачи" value={taskLabel} />}
      </dl>
      <span className="mt-4 inline-block border-b border-transparent font-sans text-xs font-medium uppercase tracking-wider text-ink-700 group-hover:border-ink-700">
        ОТКРЫТЬ АНАЛИЗ →
      </span>
    </Link>
  );
}

function CounterCard({
  label,
  count,
  loading,
  to,
  actionLabel,
  unitWord,
  unitWordPlural,
  unitWordPluralMany,
}: {
  label: string;
  count: number | null;
  loading: boolean;
  to: string;
  actionLabel: string;
  unitWord: string;
  unitWordPlural: string;
  unitWordPluralMany: string;
}) {
  if (loading || count === null) {
    return <CardSkeleton label={label} />;
  }
  const word = pluralizeRu(count, unitWord, unitWordPlural, unitWordPluralMany);
  return (
    <Link
      to={to}
      className="group border border-paper-300 bg-paper-50 p-6 transition-colors hover:border-ink-700"
    >
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <p className="mt-2 font-serif text-[3rem] font-bold leading-none tracking-tight text-paper-900">
        {formatNumber(count)}
      </p>
      <p className="mt-2 font-sans text-sm text-paper-600">{word}</p>
      <span className="mt-4 inline-block border-b border-transparent font-sans text-xs font-medium uppercase tracking-wider text-ink-700 group-hover:border-ink-700">
        {actionLabel}
      </span>
    </Link>
  );
}

function CardSkeleton({ label }: { label: string }) {
  return (
    <div className="border border-paper-300 bg-paper-50 p-6">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <div className="mt-3 h-8 w-24 animate-pulse bg-paper-200" />
      <div className="mt-3 h-3 w-32 animate-pulse bg-paper-200" />
    </div>
  );
}

function DashRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="text-[0.6875rem] uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className={`text-paper-700 ${mono ? "font-mono" : "font-sans"}`}>
        {value}
      </dd>
    </div>
  );
}

// Простая русская плюрализация: 1 анализ / 2 анализа / 5 анализов.
// Локально, без i18n-библиотеки — используется только здесь.
function pluralizeRu(
  n: number,
  one: string,
  few: string,
  many: string,
): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return `${formatNumber(n)} ${one}`;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14))
    return `${formatNumber(n)} ${few}`;
  return `${formatNumber(n)} ${many}`;
}

function GuestLanding() {
  return (
    <div className="mx-auto max-w-[1100px] px-8 py-16 lg:px-16">
      {/* Hero */}
      <section className="border-b border-paper-300 pb-12">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА · МГТУ ИМ. Н.Э. БАУМАНА · ИУ5
        </p>
        <h1 className="mt-3 max-w-3xl font-serif text-[3rem] font-bold leading-[1.05] tracking-tight text-paper-900">
          Анализатор
        </h1>
        <p className="mt-4 max-w-2xl font-serif text-[1.0625rem] leading-relaxed text-paper-600">
          Интеллектуальная система анализа наборов данных для решения задач
          машинного обучения. Загрузите CSV или XLSX — получите рекомендации по
          типу задачи, проверки качества данных и подбор похожих датасетов.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            to="/register"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            СОЗДАТЬ АККАУНТ
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center border border-ink-700 bg-paper-50 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50"
          >
            ВОЙТИ
          </Link>
        </div>
      </section>

      {/* Demo specimen */}
      <section className="mt-12">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ПРИМЕР АНАЛИЗА · ДАТАСЕТ IRIS · ДЕМОНСТРАЦИЯ
        </p>
        <div className="mt-4 border border-paper-300 bg-paper-50 px-8 py-10 lg:px-12">
          <IrisDemo />
        </div>
      </section>
    </div>
  );
}

// Статичный «срез» страницы анализа на Iris. Данные захардкожены —
// никаких API-вызовов, никакого state, не имитирует polling или interactive.
// Соответствует визуальному языку реальной AnalysisResult.tsx.
function IrisDemo() {
  return (
    <>
      <div>
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ОТЧЁТ ОБ АНАЛИЗЕ ДАТАСЕТА
        </p>
        <h2 className="mt-2 font-serif text-[1.75rem] font-bold leading-tight tracking-tight text-paper-900">
          Iris.csv
        </h2>
        <dl className="mt-4 grid grid-cols-1 gap-x-8 gap-y-1 border-y border-paper-300 py-3 sm:grid-cols-[10rem_1fr_10rem_1fr]">
          <SpecimenField label="ID анализа" value="01b3-iris-demo" />
          <SpecimenField label="Статус" value="DONE" />
          <SpecimenField label="Завершён" value="2026-05-06, 12:00" />
          <SpecimenField label="Target" value="«species»" />
        </dl>
      </div>

      <div className="mt-10 space-y-12">
        <DemoSection number={1} title="Профайл данных">
          <div className="border border-paper-300 bg-paper-50">
            <dl className="divide-y divide-paper-200">
              <DemoRow label="Имя файла" value="Iris.csv" />
              <DemoRow label="Строк × столбцов" value="150 × 5" />
              <DemoRow label="Размер в памяти" value="0.01 МБ" />
              <DemoRow label="Типы колонок" value="float64 4 · object 1" />
              <DemoRow label="Пропусков (всего)" value="0.00%" />
              <DemoRow label="Дубликатов строк" value="0.00%" />
              <DemoRow
                label="Целевая переменная"
                value={
                  <>
                    <span className="font-mono">«species»</span>
                    <span className="ml-2 text-paper-500">— категориальная</span>
                  </>
                }
                mono={false}
              />
            </dl>
          </div>
        </DemoSection>

        <DemoSection number={4} title="Рекомендация типа задачи">
          <blockquote className="border-l-[3px] border-ink-700 pl-6">
            <p className="font-serif text-[2rem] font-semibold leading-tight tracking-tight text-ink-900">
              Многоклассовая классификация
            </p>
            <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
              <span className="text-paper-500">
                CONFIDENCE{" "}
                <span className="ml-1 font-mono normal-case tracking-normal text-success-700">
                  0.950
                </span>
                <span className="ml-1 font-mono normal-case tracking-normal text-paper-500">
                  (95%)
                </span>
              </span>
              <span className="text-paper-500">
                SOURCE{" "}
                <span className="ml-1 font-mono normal-case tracking-normal text-ink-700">
                  rules
                </span>
              </span>
            </div>
          </blockquote>
        </DemoSection>
      </div>
    </>
  );
}

function DemoSection({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <header className="mb-5 border-b border-paper-300 pb-2">
        <h3 className="flex items-baseline gap-3 font-serif text-[1.375rem] font-semibold leading-snug tracking-tight text-paper-900">
          <span className="font-sans text-base font-medium text-paper-400">
            §{number}
          </span>
          {title}
        </h3>
      </header>
      {children}
    </section>
  );
}

function DemoRow({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="grid grid-cols-[12rem_1fr] gap-6 px-5 py-3">
      <dt className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className={`text-sm text-paper-800 ${mono ? "font-mono" : "font-sans"}`}>
        {value}
      </dd>
    </div>
  );
}

function SpecimenField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-3 py-1">
      <dt className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd className="font-mono text-xs text-paper-800">{value}</dd>
    </div>
  );
}
