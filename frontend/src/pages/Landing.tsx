// Главная страница — диспатчер по auth-статусу.
//
// Гость → GuestLanding (Sprint 6 Phase 6, feature tour поверх Iris-фикстуры).
// Авторизован → Dashboard (Sprint 6 Phase 7, личная панель в pages/Dashboard.tsx).
//
// До Phase 7 dashboard жил inline в этом файле (~300 строк хелперов).
// Phase 7 вынесла его в pages/Dashboard.tsx, здесь остался только роутинг.
import { ArrowDown, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { DatasetSummary } from "../components/analysis/DatasetSummary";
import { QualityFlags } from "../components/analysis/QualityFlags";
import { TaskRecommendationCard } from "../components/analysis/TaskRecommendationCard";
import {
  irisAnalysis,
  irisDataset,
  irisResult,
} from "../demo/iris";
import { Dashboard } from "./Dashboard";

export function Landing() {
  const user = useAuthStore((s) => s.user);

  if (user) {
    return <Dashboard />;
  }

  return <GuestLanding />;
}

function GuestLanding() {
  return (
    <div className="bg-paper-50">
      <Hero />
      <IntroBand />
      <AnalysisHeaderBand />
      <Section1Recommendation />
      <Section2DatasetSummary />
      <Section3QualityFlags />
      <Section4Distributions />
      <Section5Similar />
      <Section6Baseline />
      <Section7Report />
      <FinalCta />
    </div>
  );
}

// Полоса на всю ширину viewport. Внутри — контейнер max-w-[1200px] с
// двухколоночным grid: основной блок слева (1fr), DemoCallout справа в
// узкой колонке 300px. На mobile (<lg) — стек: блок, под ним подпись.
function Band({
  tone,
  children,
  className = "",
}: {
  tone: "50" | "100";
  children: React.ReactNode;
  className?: string;
}) {
  const bg = tone === "100" ? "bg-paper-100" : "bg-paper-50";
  return (
    <section className={bg}>
      <div
        className={`mx-auto max-w-[1200px] px-8 lg:px-16 ${className || "py-12 lg:py-14"}`}
      >
        {children}
      </div>
    </section>
  );
}

// Внутренний layout band'а: «контент + подпись справа на широких».
function BandLayout({
  children,
  callout,
  calloutTone,
}: {
  children: React.ReactNode;
  callout: React.ReactNode;
  /** Цвет фона callout — контрастный к band'у. */
  calloutTone: "50" | "100";
}) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_300px] lg:gap-10">
      <div className="min-w-0">{children}</div>
      <DemoCallout tone={calloutTone}>{callout}</DemoCallout>
    </div>
  );
}

// Подпись-«рубрика» рядом с блоком. Левая info-граница + контрастный фон
// (paper-50 над paper-100 band и наоборот). Plex Sans, не отвлекает от
// основного блока, но ясно видна как отдельный пояснительный голос.
function DemoCallout({
  tone,
  children,
}: {
  tone: "50" | "100";
  children: React.ReactNode;
}) {
  const bg = tone === "100" ? "bg-paper-100" : "bg-paper-50";
  return (
    <aside
      className={`border-l-2 border-info-500 ${bg} p-4 font-sans text-sm leading-relaxed text-paper-700`}
    >
      <p className="mb-1.5 font-medium uppercase tracking-wider text-[0.6875rem] text-info-700">
        Что здесь происходит
      </p>
      {children}
    </aside>
  );
}

// Заголовок секции в стиле AnalysisResult: «§N · Название».
function SectionHeader({
  number,
  title,
}: {
  number: number;
  title: string;
}) {
  return (
    <header className="mb-4 border-b border-paper-300 pb-2">
      <h2 className="flex items-baseline gap-3 font-serif text-[1.5rem] font-semibold leading-snug tracking-tight text-paper-900">
        <span className="font-sans text-base font-medium text-paper-400">
          §{number}
        </span>
        {title}
      </h2>
    </header>
  );
}

function Hero() {
  return (
    <section className="bg-paper-50">
      <div className="mx-auto flex max-w-[800px] flex-col items-center px-6 py-20 text-center lg:py-28">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-600">
          ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА · МГТУ ИМ. Н.Э. БАУМАНА · ИУ5
        </p>
        <h1 className="mt-6 font-serif text-[3.5rem] font-bold uppercase leading-[1.0] tracking-tight text-paper-900 sm:text-[4.5rem] lg:text-[5.5rem]">
          Анализатор
        </h1>
        <p className="mt-6 max-w-prose font-serif text-[1.0625rem] leading-relaxed text-paper-700 lg:text-[1.125rem]">
          Интеллектуальная система анализа наборов данных для решения задач
          машинного обучения. Загрузите CSV или XLSX — получите рекомендации
          по типу задачи, проверки качества данных и подбор похожих
          датасетов.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            to="/register"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            СОЗДАТЬ АККАУНТ
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center border border-paper-400 bg-paper-50 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-700 transition-colors hover:border-ink-700 hover:text-ink-700"
          >
            ВОЙТИ
          </Link>
        </div>
        <a
          href="#feature-tour"
          className="mt-16 inline-flex items-center gap-1.5 font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500 underline-offset-4 hover:text-ink-700 hover:underline"
        >
          НИЖЕ — ПРИМЕР РАБОТЫ
          <ArrowDown className="h-3.5 w-3.5" aria-hidden />
        </a>
      </div>
    </section>
  );
}

function IntroBand() {
  return (
    <Band tone="100" className="py-14 lg:py-20">
      <div id="feature-tour" className="mx-auto max-w-[760px] text-center">
        <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
          ВИТРИНА · РАЗДЕЛЫ §1–§7
        </p>
        <h2 className="mt-2 font-serif text-[2rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[2.5rem]">
          Как это работает
        </h2>
        <p className="mt-5 font-serif text-[1rem] leading-relaxed text-paper-700 lg:text-[1.0625rem]">
          Ниже — пример анализа классического датасета Iris (150 строк, 5
          колонок, целевая переменная — вид цветка). Под каждым блоком —
          описание того, что система делает в этом месте.
        </p>
      </div>
    </Band>
  );
}

// Шапка анализа в стиле AnalysisResult.tsx (компактная meta-row под
// именем файла). Использует данные из demo-фикстуры.
function AnalysisHeaderBand() {
  const finishedAt = irisAnalysis.finished_at
    ? new Date(irisAnalysis.finished_at).toLocaleString("ru-RU")
    : null;
  return (
    <Band tone="50" className="py-12 lg:py-14">
      <BandLayout
        calloutTone="100"
        callout={
          <p>
            Перед запуском анализа пользователь может указать целевую
            переменную (target). Здесь это{" "}
            <span className="font-mono">«species»</span> — колонка, которую
            система будет учить предсказывать.
          </p>
        }
      >
        <div>
          <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
            ОТЧЁТ ОБ АНАЛИЗЕ
          </p>
          <h2 className="mt-2 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[2.5rem]">
            {irisDataset.original_filename}
          </h2>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
            <MetaPair label="Статус" value="DONE" tone="text-success-700" />
            {irisAnalysis.target_column && (
              <MetaPair
                label="Target"
                value={`«${irisAnalysis.target_column}»`}
              />
            )}
            {finishedAt && <MetaPair label="Завершён" value={finishedAt} />}
          </div>
        </div>
      </BandLayout>
    </Band>
  );
}

function MetaPair({
  label,
  value,
  tone = "text-ink-700",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <span className="text-paper-500">
      {label.toUpperCase()}{" "}
      <span className={`ml-1 font-mono normal-case tracking-normal ${tone}`}>
        {value}
      </span>
    </span>
  );
}

function Section1Recommendation() {
  return (
    <Band tone="100">
      <SectionHeader number={1} title="Рекомендация типа задачи" />
      <BandLayout
        calloutTone="50"
        callout={
          <p>
            Система определяет тип ML-задачи по двум слоям: правила (Слой 1)
            для очевидных случаев и meta-классификатор (Слой 2) для
            пограничных. Объяснение помогает понять, почему было принято
            именно такое решение.
          </p>
        }
      >
        <TaskRecommendationCard
          recommendation={irisResult.task_recommendation}
        />
      </BandLayout>
    </Band>
  );
}

function Section2DatasetSummary() {
  return (
    <Band tone="50">
      <SectionHeader number={2} title="Сводка датасета" />
      <BandLayout
        calloutTone="100"
        callout={
          <p>
            Базовые метрики качества данных: размерность, полнота, типы
            признаков. Сразу видно, есть ли пропуски и дубликаты, нужна ли
            предобработка.
          </p>
        }
      >
        <DatasetSummary meta={irisResult.meta_features} />
      </BandLayout>
    </Band>
  );
}

function Section3QualityFlags() {
  return (
    <Band tone="100">
      <SectionHeader number={3} title="Качество данных" />
      <BandLayout
        calloutTone="50"
        callout={
          <p>
            Двенадцать правил качества проверяют датасет на типичные
            проблемы: пропуски, выбросы, кардинальность, утечки целевой
            переменной. Каждое замечание раскрывается с деталями и
            рекомендациями.
          </p>
        }
      >
        <QualityFlags flags={irisResult.flags} />
      </BandLayout>
    </Band>
  );
}

function Section4Distributions() {
  // Один реалистичный график вместо полного Distributions: распределение
  // целевой переменной по 3 классам (по 50 экземпляров каждого вида —
  // классическая структура Iris). Inline-SVG, без зависимости от Plotly.
  const counts = [
    { label: "setosa", n: 50 },
    { label: "versicolor", n: 50 },
    { label: "virginica", n: 50 },
  ];
  const max = Math.max(...counts.map((c) => c.n));
  return (
    <Band tone="50">
      <SectionHeader number={4} title="Распределения" />
      <BandLayout
        calloutTone="100"
        callout={
          <p>
            Гистограммы признаков, распределение целевой переменной и
            матрица корреляций между числовыми признаками. Помогает заметить
            асимметрию, мультимодальность, скрытые группы.
          </p>
        }
      >
        <figure className="border border-paper-300 bg-paper-50 p-6">
          <p className="mb-4 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
            ЦЕЛЕВАЯ ПЕРЕМЕННАЯ · SPECIES
          </p>
          <svg
            viewBox="0 0 480 200"
            className="h-auto w-full font-mono"
            role="img"
            aria-label="Распределение целевой переменной species: 50 setosa, 50 versicolor, 50 virginica"
          >
            {/* Базовая ось */}
            <line
              x1="60"
              y1="170"
              x2="460"
              y2="170"
              stroke="#9F9583"
              strokeWidth="1"
            />
            {counts.map((c, i) => {
              const barW = 100;
              const x = 80 + i * 130;
              const h = (c.n / max) * 130;
              const y = 170 - h;
              return (
                <g key={c.label}>
                  <rect
                    x={x}
                    y={y}
                    width={barW}
                    height={h}
                    fill="#1F1B16"
                    opacity="0.85"
                  />
                  <text
                    x={x + barW / 2}
                    y={y - 6}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#1F1B16"
                  >
                    {c.n}
                  </text>
                  <text
                    x={x + barW / 2}
                    y={188}
                    textAnchor="middle"
                    fontSize="11"
                    fill="#5D5648"
                  >
                    {c.label}
                  </text>
                </g>
              );
            })}
          </svg>
          <figcaption className="mt-3 font-sans text-xs text-paper-500">
            Рис. 1. Распределение целевого признака{" "}
            <span className="font-mono">species</span> — 3 класса по 50
            наблюдений. Сбалансированный multi-class.
          </figcaption>
        </figure>
      </BandLayout>
    </Band>
  );
}

function Section5Similar() {
  return (
    <Band tone="100">
      <SectionHeader number={5} title="Похожие датасеты" />
      <BandLayout
        calloutTone="50"
        callout={
          <p>
            Каталог из 29 публичных датасетов с UCI / sklearn / GitHub.
            Помогает найти похожие задачи и подсмотреть, какие модели у них
            хорошо работают.
          </p>
        }
      >
        <div>
          <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
            Top-5 близких записей из каталога. Сходство — косинусная мера в
            пространстве 16 + 1 мета-признаков, индекс HNSW. Пример первой
            строки результата:
          </p>
          <ol className="mt-4 border-y border-paper-200">
            <li className="grid grid-cols-[2rem_1fr_auto] items-start gap-4 px-1 py-3">
              <span className="pt-0.5 font-mono text-sm text-paper-400">
                01.
              </span>
              <div className="min-w-0">
                <p className="truncate font-serif text-[1rem] font-semibold leading-snug text-paper-900">
                  Wine Quality (Red)
                </p>
                <p className="mt-1 font-mono text-xs text-paper-500">
                  multiclass classification · UCI · cos d{" "}
                  <span className="text-paper-700">0.755</span>
                </p>
              </div>
              <span className="border border-paper-400 px-2 py-0.5 font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-600">
                CATALOG
              </span>
            </li>
          </ol>
        </div>
      </BandLayout>
    </Band>
  );
}

function Section6Baseline() {
  return (
    <Band tone="50">
      <SectionHeader number={6} title="Базовая модель" />
      <BandLayout
        calloutTone="100"
        callout={
          <p>
            Не для production-использования. Цель baseline — быстро понять,
            есть ли в данных полезный сигнал и стоит ли вкладываться в более
            сложную модель.
          </p>
        }
      >
        <div>
          <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
            Система обучает две baseline-модели (линейную и Random Forest)
            с 5-fold кросс-валидацией и автоматически исключает колонки с
            подозрением на утечку. Для классификации считается accuracy /
            F1 / ROC-AUC, для регрессии — R² / MAE / RMSE. По метрикам
            формируется вердикт о качестве сигнала в данных.
          </p>
          <div className="mt-5 border-l-[3px] border-success-500 bg-paper-50 px-5 py-4">
            <p className="font-sans text-xs font-medium uppercase tracking-wider text-success-700">
              ВЕРДИКТ · УВЕРЕННЫЙ СИГНАЛ
            </p>
            <p className="mt-1.5 font-serif text-[0.9375rem] text-paper-700">
              Модель уверенно справляется с задачей. Признаки несут
              достаточно информации о целевой переменной.
            </p>
          </div>
        </div>
      </BandLayout>
    </Band>
  );
}

function Section7Report() {
  return (
    <Band tone="100">
      <SectionHeader number={7} title="PDF-отчёт" />
      <BandLayout
        calloutTone="50"
        callout={
          <p>
            Отчёт сохраняется в библиотеке пользователя — можно скачать
            снова в любой момент или сравнить версии при повторных
            анализах.
          </p>
        }
      >
        <div className="flex items-start gap-5">
          <div className="hidden shrink-0 sm:block">
            <div className="flex h-20 w-16 items-center justify-center border border-paper-400 bg-paper-50">
              <FileText
                className="h-8 w-8 text-paper-500"
                aria-hidden
                strokeWidth={1.25}
              />
            </div>
          </div>
          <div className="min-w-0">
            <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-700">
              Все результаты анализа собираются в один PDF-документ
              (15–20 страниц): сводка, флаги качества, графики,
              рекомендация и метрики baseline. Удобно прикладывать к
              отчётам или делиться с командой.
            </p>
          </div>
        </div>
      </BandLayout>
    </Band>
  );
}

function FinalCta() {
  return (
    <section className="bg-paper-100">
      <div className="mx-auto flex max-w-[760px] flex-col items-center px-8 py-16 text-center lg:py-24">
        <h2 className="font-serif text-[2rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[2.5rem]">
          Готовы попробовать на своих данных?
        </h2>
        <p className="mt-4 max-w-prose font-serif text-[1rem] leading-relaxed text-paper-700">
          Зарегистрируйтесь, чтобы загрузить свой датасет и получить полный
          анализ.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            to="/register"
            className="inline-flex items-center border border-ink-700 bg-ink-700 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800"
          >
            СОЗДАТЬ АККАУНТ
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center border border-paper-400 bg-paper-50 px-5 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-700 transition-colors hover:border-ink-700 hover:text-ink-700"
          >
            ВОЙТИ
          </Link>
        </div>
      </div>
    </section>
  );
}

