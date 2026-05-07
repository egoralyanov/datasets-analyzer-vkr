// Страница результата анализа.
//
// Sprint 6, Phase 2.5: layout refinement —
// 1) Рекомендация типа задачи поднята в самый верх как hero (§1).
// 2) Шапка анализа компактная: имя файла h1 + meta-row STATUS · TARGET ·
//    ЗАВЕРШЁН в одну строку. UUID-а ID анализа в UI больше нет (он есть в
//    URL — этого достаточно для технических нужд).
// 3) Каждая секция — отдельная горизонтальная полоса на всю ширину viewport
//    с чередующимся фоном paper-50 / paper-100. Между полосами нет gap'а —
//    они прилегают, фон плавно меняется как полосы в журнале.
// 4) Нет двухколоночного layout'а Качество + Рекомендация — обе секции
//    full-width.
// 5) Вертикальные отступы внутри секций уменьшены ~30%: py-10 на bands,
//    space-y-* внутри тоже стянуто.
//
// State для baseline и report лифтнут через useBaselineActions /
// useReportActions, sticky-bar и карточки делят один статус.
//
// См. frontend/DESIGN_TOKENS.md, разделы 3 и 8.4 + plans/06-...md, Phase 2.5.
import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { analysesApi } from "../api/analyses";
import { datasetsApi } from "../api/datasets";
import { useAnalysisPolling } from "../hooks/useAnalysisPolling";
import { useBaselineActions } from "../hooks/useBaselineActions";
import { useReportActions } from "../hooks/useReportActions";
import { DatasetSummary } from "../components/analysis/DatasetSummary";
import { QualityFlags } from "../components/analysis/QualityFlags";
import { Distributions } from "../components/analysis/Distributions";
import { TaskRecommendationCard } from "../components/analysis/TaskRecommendationCard";
import { SimilarDatasetsCard } from "../components/analysis/SimilarDatasetsCard";
import { BaselineCard } from "../components/analysis/BaselineCard";
import { ReportDownloadCard } from "../components/analysis/ReportDownloadCard";
import {
  InlineActionBar,
  StickyActionBar,
} from "../components/analysis/StickyActionBar";
import type { AnalysisStatus } from "../types/analysis";

export function AnalysisResult() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // Polling /analyses/{id} каждые 2 секунды, пока pending/running.
  const polling = useAnalysisPolling(id);
  const analysis = polling.data;

  // Когда status стал done — тянем полный результат с meta_features и flags.
  const result = useQuery({
    queryKey: ["analysisResult", id],
    queryFn: () => analysesApi.getResult(id!),
    enabled: !!id && analysis?.status === "done",
  });

  // Имя файла нужно для shapки страницы и sticky-bar.
  const dataset = useQuery({
    queryKey: ["dataset", analysis?.dataset_id],
    queryFn: () => datasetsApi.get(analysis!.dataset_id),
    enabled: !!analysis?.dataset_id,
  });

  // Lifted state: baseline / report. Хуки безопасно работают с undefined id —
  // polling guard'ится через `enabled`, mutation проверяет id перед вызовом.
  const baselineActions = useBaselineActions(id);
  const reportActions = useReportActions(id);

  // Скролл вверх при смене анализа.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [id]);

  if (!id) {
    return (
      <Band tone="50">
        <ErrorBox message="Идентификатор анализа отсутствует в URL." />
      </Band>
    );
  }
  if (polling.isLoading || !analysis) {
    return (
      <Band tone="50">
        <SpinnerBox label="Загрузка анализа…" />
      </Band>
    );
  }
  if (polling.isError) {
    return (
      <Band tone="50">
        <ErrorBox message="Не удалось получить статус анализа." />
      </Band>
    );
  }

  const filename = dataset.data?.original_filename ?? "Анализ";
  const finishedAt = analysis.finished_at
    ? new Date(analysis.finished_at).toLocaleString("ru-RU")
    : null;
  const showSticky = analysis.status === "done" && !!result.data;

  return (
    <>
      <Band tone="50" className="pt-10 pb-6">
        <button
          type="button"
          onClick={() => navigate("/upload")}
          className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500 underline-offset-2 hover:text-ink-700 hover:underline"
        >
          ← НАЗАД К ДАТАСЕТАМ
        </button>
        <PageHeader
          filename={filename}
          status={analysis.status}
          target={analysis.target_column}
          finishedAt={finishedAt}
        />
      </Band>

      {(analysis.status === "pending" || analysis.status === "running") && (
        <Band tone="100">
          <RunningView />
        </Band>
      )}

      {analysis.status === "failed" && (
        <Band tone="100">
          <FailedView errorMessage={analysis.error_message} />
        </Band>
      )}

      {analysis.status === "done" && result.data && (
        <>
          <Band tone="100">
            <Section number={1} title="Рекомендация типа задачи">
              <TaskRecommendationCard
                recommendation={result.data.task_recommendation}
              />
            </Section>
          </Band>

          <Band tone="50">
            <Section number={2} title="Сводка датасета">
              <DatasetSummary meta={result.data.meta_features} />
            </Section>
          </Band>

          <Band tone="100">
            <Section
              number={3}
              title="Качество данных"
              note={
                result.data.flags.length > 0
                  ? `${result.data.flags.length} замечаний`
                  : "без замечаний"
              }
            >
              <QualityFlags flags={result.data.flags} />
            </Section>
          </Band>

          <Band tone="50">
            <Section number={4} title="Распределения">
              <Distributions meta={result.data.meta_features} />
            </Section>
          </Band>

          <Band tone="100">
            <Section number={5} title="Похожие датасеты">
              <SimilarDatasetsCard analysisId={id} />
            </Section>
          </Band>

          <Band tone="50">
            <Section number={6} title="Базовая модель">
              <BaselineCard
                taskType={result.data.task_recommendation?.task_type_code}
                actions={baselineActions}
              />
            </Section>
          </Band>

          <Band tone="100" lastBeforeSticky>
            <Section number={7} title="PDF-отчёт">
              <ReportDownloadCard
                analysisStatus={analysis.status}
                actions={reportActions}
              />
            </Section>
            <InlineActionBar
              analysisStatus={analysis.status}
              baseline={baselineActions}
              report={reportActions}
            />
          </Band>
        </>
      )}

      {analysis.status === "done" && result.isLoading && (
        <Band tone="100">
          <SpinnerBox label="Загрузка результата…" />
        </Band>
      )}

      {showSticky && (
        <StickyActionBar
          filename={filename}
          analysisStatus={analysis.status}
          baseline={baselineActions}
          report={reportActions}
        />
      )}
    </>
  );
}

// Полоса фона на всю ширину viewport. Контент внутри ограничен max-w-[1200px].
// `tone` — paper-50 / paper-100; чередование задаёт визуальное членение
// страницы как в журнале. Никаких теней и скруглений.
function Band({
  tone,
  children,
  className = "",
  lastBeforeSticky = false,
}: {
  tone: "50" | "100";
  children: React.ReactNode;
  className?: string;
  /** Доп. отступ снизу, чтобы sticky-bar не перекрывал контент на lg+. */
  lastBeforeSticky?: boolean;
}) {
  const bg = tone === "100" ? "bg-paper-100" : "bg-paper-50";
  const padBase = className || "py-10";
  const padBottom = lastBeforeSticky ? "lg:pb-24" : "";
  return (
    <section className={bg}>
      <div
        className={`mx-auto max-w-[1200px] px-8 lg:px-16 ${padBase} ${padBottom}`}
      >
        {children}
      </div>
    </section>
  );
}

function PageHeader({
  filename,
  status,
  target,
  finishedAt,
}: {
  filename: string;
  status: AnalysisStatus;
  target: string | null;
  finishedAt: string | null;
}) {
  return (
    <div className="mt-5">
      <h1 className="font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900 sm:text-[2.5rem]">
        {filename}
      </h1>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-sans text-xs uppercase tracking-wider">
        <MetaPair label="Статус" value={status.toUpperCase()} tone={statusTone(status)} mono />
        {target && (
          <MetaPair label="Target" value={`«${target}»`} mono />
        )}
        {finishedAt && (
          <MetaPair label="Завершён" value={finishedAt} mono />
        )}
      </div>
    </div>
  );
}

function statusTone(status: AnalysisStatus): string {
  switch (status) {
    case "done":
      return "text-success-700";
    case "failed":
      return "text-critical-700";
    case "running":
      return "text-info-700";
    case "pending":
    default:
      return "text-paper-600";
  }
}

function MetaPair({
  label,
  value,
  tone = "text-ink-700",
  mono = false,
}: {
  label: string;
  value: string;
  tone?: string;
  mono?: boolean;
}) {
  return (
    <span className="text-paper-500">
      {label.toUpperCase()}{" "}
      <span
        className={`ml-1 normal-case tracking-normal ${tone} ${mono ? "font-mono" : "font-sans"}`}
      >
        {value}
      </span>
    </span>
  );
}

function Section({
  number,
  title,
  note,
  children,
}: {
  number: number;
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <header className="mb-4 flex items-baseline justify-between gap-4 border-b border-paper-300 pb-2">
        <h2 className="flex items-baseline gap-3 font-serif text-[1.5rem] font-semibold leading-snug tracking-tight text-paper-900">
          <span className="font-sans text-base font-medium text-paper-400">
            §{number}
          </span>
          {title}
        </h2>
        {note && (
          <span className="font-sans text-xs uppercase tracking-wider text-paper-500">
            {note}
          </span>
        )}
      </header>
      {children}
    </section>
  );
}

function RunningView() {
  return (
    <div className="border border-paper-300 bg-paper-50 px-12 py-16">
      <div className="flex flex-col items-center gap-4 text-center">
        <Loader2 className="h-8 w-8 animate-spin text-ink-700" />
        <p className="font-serif text-[1.25rem] font-semibold text-paper-900">
          Идёт анализ…
        </p>
        <p className="max-w-md font-serif text-sm leading-relaxed text-paper-600">
          Профайлер вычисляет meta-features, после чего применяются 12 правил
          качества. Обычно это занимает 5–30 секунд.
        </p>
      </div>
    </div>
  );
}

function FailedView({ errorMessage }: { errorMessage: string | null }) {
  const navigate = useNavigate();
  return (
    <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-5 py-4">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-critical-700">
        FAIL · АНАЛИЗ ЗАВЕРШИЛСЯ С ОШИБКОЙ
      </p>
      <p className="mt-2 font-serif text-[0.9375rem] leading-relaxed text-paper-700">
        {errorMessage || "Внутренняя ошибка сервера. Попробуйте ещё раз."}
      </p>
      <button
        type="button"
        onClick={() => navigate("/upload")}
        className="mt-3 inline-flex items-center gap-2 border border-ink-700 bg-paper-50 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50"
      >
        ← К ДАТАСЕТАМ
      </button>
    </div>
  );
}

function SpinnerBox({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-8 font-sans text-sm text-paper-600">
      <Loader2 className="h-5 w-5 animate-spin text-ink-700" />
      <span>{label}</span>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-3 font-serif text-sm text-paper-700">
      {message}
    </div>
  );
}
