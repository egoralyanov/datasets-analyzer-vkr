// Страница результата анализа.
//
// Sprint 6, Phase 2: убрана линейная «простыня» — макет ближе к dashboard
// научной публикации. Сводка сжата до 5-карточечной строки + dtype-bar
// (§1). Качество и Рекомендация — две колонки на ≥1280px (§2). Распределения
// в 2x2 (§3). Похожие — компактный каталог (§4). Базовая модель + важность
// признаков рядом (§5). PDF-отчёт — отдельная секция (§6) и shortcut в
// sticky-bar / inline-bar в зависимости от viewport.
//
// State для baseline и report лифтнут наверх через `useBaselineActions` и
// `useReportActions` — sticky-bar и карточки делят один статус.
//
// См. frontend/DESIGN_TOKENS.md, разделы 3 и 8.4 + plans/06-...md, Phase 2.
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
    return <ErrorBox message="Идентификатор анализа отсутствует в URL." />;
  }
  if (polling.isLoading || !analysis) {
    return <SpinnerBox label="Загрузка анализа…" />;
  }
  if (polling.isError) {
    return <ErrorBox message="Не удалось получить статус анализа." />;
  }

  const finishedAt = analysis.finished_at
    ? new Date(analysis.finished_at).toLocaleString("ru-RU")
    : null;
  const filename = dataset.data?.original_filename ?? "Анализ";
  const showSticky = analysis.status === "done" && !!result.data;

  return (
    <>
      {/* pb-24 — отступ под высоту sticky-bar, чтобы он не перекрывал контент. */}
      <div className="mx-auto max-w-[1200px] px-8 py-12 lg:px-16 lg:pb-24">
        <button
          type="button"
          onClick={() => navigate("/upload")}
          className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500 underline-offset-2 hover:text-ink-700 hover:underline"
        >
          ← НАЗАД К ДАТАСЕТАМ
        </button>

        <div className="mt-6">
          <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
            ОТЧЁТ ОБ АНАЛИЗЕ ДАТАСЕТА
          </p>
          <h1 className="mt-2 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
            {filename}
          </h1>
          <dl className="mt-5 grid grid-cols-1 gap-x-8 gap-y-1 border-y border-paper-300 py-3 sm:grid-cols-[10rem_1fr_10rem_1fr]">
            <SpecimenField label="ID анализа" value={analysis.id} mono />
            <SpecimenField
              label="Статус"
              value={analysis.status.toUpperCase()}
              mono
            />
            {finishedAt && (
              <SpecimenField label="Завершён" value={finishedAt} mono />
            )}
            {analysis.target_column && (
              <SpecimenField
                label="Target"
                value={`«${analysis.target_column}»`}
                mono
              />
            )}
          </dl>
        </div>

        {(analysis.status === "pending" || analysis.status === "running") && (
          <RunningView />
        )}

        {analysis.status === "failed" && (
          <FailedView errorMessage={analysis.error_message} />
        )}

        {analysis.status === "done" && result.data && (
          <div className="mt-10 space-y-12">
            <Section number={1} title="Сводка датасета">
              <DatasetSummary meta={result.data.meta_features} />
            </Section>

            <div className="grid gap-10 xl:grid-cols-2 xl:gap-8">
              <Section
                number={2}
                title="Качество данных"
                note={
                  result.data.flags.length > 0
                    ? `${result.data.flags.length} замечаний`
                    : "без замечаний"
                }
              >
                <QualityFlags flags={result.data.flags} />
              </Section>

              <Section number={3} title="Рекомендация типа задачи">
                <TaskRecommendationCard
                  recommendation={result.data.task_recommendation}
                />
              </Section>
            </div>

            <Section number={4} title="Распределения">
              <Distributions meta={result.data.meta_features} />
            </Section>

            <Section number={5} title="Похожие датасеты">
              <SimilarDatasetsCard analysisId={id} />
            </Section>

            <Section number={6} title="Базовая модель">
              <BaselineCard
                taskType={result.data.task_recommendation?.task_type_code}
                actions={baselineActions}
              />
            </Section>

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
          </div>
        )}

        {analysis.status === "done" && result.isLoading && (
          <SpinnerBox label="Загрузка результата…" />
        )}
      </div>

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
      <header className="mb-5 flex items-baseline justify-between gap-4 border-b border-paper-300 pb-2">
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

function SpecimenField({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline gap-3 py-1">
      <dt className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd
        className={`text-sm text-paper-800 ${
          mono ? "font-mono text-xs" : "font-sans"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

function RunningView() {
  return (
    <div className="mt-8 border border-paper-300 bg-paper-50 px-12 py-16">
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
    <div className="mt-8 border-l-[3px] border-critical-500 bg-critical-50/70 px-5 py-4">
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
    <div className="mx-auto max-w-[1200px] px-8 py-16 lg:px-16">
      <div className="flex items-center justify-center gap-3 border border-paper-300 bg-paper-50 p-8 font-sans text-sm text-paper-600">
        <Loader2 className="h-5 w-5 animate-spin text-ink-700" />
        <span>{label}</span>
      </div>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="mx-auto max-w-[1200px] px-8 py-16 lg:px-16">
      <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-3 font-serif text-sm text-paper-700">
        {message}
      </div>
    </div>
  );
}
