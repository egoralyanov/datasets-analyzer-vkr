import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Loader2, XCircle } from "lucide-react";
import { analysesApi } from "../api/analyses";
import { datasetsApi } from "../api/datasets";
import { useAnalysisPolling } from "../hooks/useAnalysisPolling";
import { DatasetSummary } from "../components/analysis/DatasetSummary";
import { QualityFlags } from "../components/analysis/QualityFlags";
import { Distributions } from "../components/analysis/Distributions";
import { TaskRecommendationCard } from "../components/analysis/TaskRecommendationCard";
import { SimilarDatasetsCard } from "../components/analysis/SimilarDatasetsCard";
import { BaselineCard } from "../components/analysis/BaselineCard";
import { ReportDownloadCard } from "../components/analysis/ReportDownloadCard";

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

  // Имя файла нужно для DatasetSummary — берём через GET /datasets/{id}.
  const dataset = useQuery({
    queryKey: ["dataset", analysis?.dataset_id],
    queryFn: () => datasetsApi.get(analysis!.dataset_id),
    enabled: !!analysis?.dataset_id,
  });

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

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <button
        type="button"
        onClick={() => navigate("/upload")}
        className="inline-flex items-center gap-1 text-sm text-slate-600 hover:text-slate-900"
      >
        <ArrowLeft className="h-4 w-4" />
        Назад к датасетам
      </button>

      <h1 className="mt-3 text-2xl font-semibold text-slate-900">
        Результат анализа
      </h1>
      <p className="mt-1 text-sm text-slate-600">
        ID анализа:{" "}
        <span className="font-mono text-xs text-slate-700">{analysis.id}</span>
      </p>

      {(analysis.status === "pending" || analysis.status === "running") && (
        <RunningView />
      )}

      {analysis.status === "failed" && (
        <FailedView errorMessage={analysis.error_message} />
      )}

      {analysis.status === "done" && result.data && (
        <div className="mt-6 space-y-6">
          <DatasetSummary
            meta={result.data.meta_features}
            filename={dataset.data?.original_filename}
            targetColumn={analysis.target_column}
          />
          <QualityFlags flags={result.data.flags} />
          <Distributions meta={result.data.meta_features} />
          <TaskRecommendationCard
            recommendation={result.data.task_recommendation}
          />
          <SimilarDatasetsCard analysisId={id} />
          <BaselineCard
            analysisId={id}
            taskType={result.data.task_recommendation?.task_type_code}
          />
          <ReportDownloadCard
            analysisId={id}
            analysisStatus={analysis.status}
          />
        </div>
      )}

      {analysis.status === "done" && result.isLoading && (
        <SpinnerBox label="Загрузка результата…" />
      )}
    </div>
  );
}

function RunningView() {
  return (
    <div className="mt-8 flex flex-col items-center gap-4 rounded-lg border border-slate-200 bg-white p-12">
      <Loader2 className="h-10 w-10 animate-spin text-blue-600" />
      <p className="text-lg font-medium text-slate-900">Идёт анализ…</p>
      <p className="text-sm text-slate-600">
        Профайлер вычисляет meta-features, после чего применятся 12 правил
        качества. Обычно это занимает 5–30 секунд.
      </p>
    </div>
  );
}

function FailedView({ errorMessage }: { errorMessage: string | null }) {
  const navigate = useNavigate();
  return (
    <div className="mt-6 rounded-lg border border-red-300 bg-red-50 p-6">
      <div className="flex items-start gap-3">
        <XCircle className="mt-0.5 h-6 w-6 shrink-0 text-red-600" />
        <div>
          <h2 className="text-lg font-semibold text-red-900">
            Анализ завершился с ошибкой
          </h2>
          <p className="mt-2 text-sm text-red-800">
            {errorMessage || "Внутренняя ошибка сервера. Попробуйте ещё раз."}
          </p>
          <button
            type="button"
            onClick={() => navigate("/upload")}
            className="mt-4 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100"
          >
            <ArrowLeft className="h-4 w-4" />К датасетам
          </button>
        </div>
      </div>
    </div>
  );
}

function SpinnerBox({ label }: { label: string }) {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="flex items-center justify-center gap-3 rounded-lg border border-slate-200 bg-white p-8 text-slate-600">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span>{label}</span>
      </div>
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        {message}
      </div>
    </div>
  );
}
