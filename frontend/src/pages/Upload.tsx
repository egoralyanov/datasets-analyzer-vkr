// Страница загрузки и управления датасетами.
//
// Стиль (Sprint 5, Phase 3.6): scientific/archive — серифные §-заголовки,
// metadata в Mono, кнопки в archive-стиле, список датасетов как нумерованный
// каталог с hairline-руллями (по аналогии с SimilarDatasetsCard).
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { datasetsApi } from "../api/datasets";
import { analysesApi } from "../api/analyses";
import type { Dataset, DatasetWithPreview } from "../types/dataset";
import { formatBytes, formatDateTime, formatNumber } from "../lib/format";
import { PLURAL_FORMS, pluralize } from "../lib/pluralize";
import { FileDropZone } from "../components/upload/FileDropZone";
import { DatasetPreview } from "../components/upload/DatasetPreview";
import { StartAnalysisModal } from "../components/upload/StartAnalysisModal";
import { DeleteDatasetModal } from "../components/upload/DeleteDatasetModal";

type Toast = { kind: "success" | "error"; text: string } | null;
type DeleteTarget = { id: string; filename: string };

export function Upload() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const previewRef = useRef<HTMLDivElement>(null);

  const [current, setCurrent] = useState<DatasetWithPreview | null>(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [toast, setToast] = useState<Toast>(null);
  const [analyzeOpen, setAnalyzeOpen] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  // Sprint 6, Phase 5.1: window.confirm заменён на DeleteDatasetModal с
  // загрузкой usage (analyses+reports counts). Open state — целевая запись
  // или null когда модалка закрыта.
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget | null>(null);

  // Авто-скрытие тоста через 3 секунды.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(t);
  }, [toast]);

  const listQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: datasetsApi.list,
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => datasetsApi.upload(file, setUploadPercent),
    onSuccess: (data) => {
      setCurrent(data);
      setUploadPercent(0);
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setToast({ kind: "success", text: "Датасет успешно загружен." });
      previewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    onError: (err) => {
      setUploadPercent(0);
      setToast({ kind: "error", text: extractDatasetError(err) });
    },
  });

  const openMutation = useMutation({
    mutationFn: (id: string) => datasetsApi.get(id),
    onSuccess: (data) => {
      setCurrent(data);
      previewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    onError: () => {
      setToast({ kind: "error", text: "Не удалось загрузить датасет." });
    },
  });

  // deleteMutation вынесен в DeleteDatasetModal. Тут только хук-callback,
  // чтобы при удалении активного датасета закрыть его превью.
  const onDeleteRequest = (id: string, name: string) => {
    setDeleteTarget({ id, filename: name });
  };
  const onDeleted = (id: string) => {
    if (current?.id === id) setCurrent(null);
  };

  const startAnalysisMutation = useMutation({
    mutationFn: (params: { target_column: string | null }) => {
      if (!current) throw new Error("Сначала выберите датасет");
      return analysesApi.start(current.id, params);
    },
    onSuccess: (analysis) => {
      setAnalyzeOpen(false);
      setAnalyzeError(null);
      navigate(`/analyses/${analysis.id}`);
    },
    onError: (err) => {
      setAnalyzeError(extractAnalysisError(err));
    },
  });

  return (
    <div className="mx-auto max-w-[1100px] px-8 py-12 lg:px-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        ЗАГРУЗКА И УПРАВЛЕНИЕ
      </p>
      <h1 className="mt-2 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
        Датасеты
      </h1>
      <p className="mt-3 max-w-2xl font-serif text-[0.9375rem] leading-relaxed text-paper-600">
        Поддерживаются CSV и XLSX. Файл сохраняется только для вашего аккаунта,
        кодировка и разделитель определяются автоматически.
      </p>

      {toast && (
        <div
          className={`mt-6 border-l-[3px] px-4 py-2 font-sans text-sm ${
            toast.kind === "success"
              ? "border-success-500 bg-paper-50 text-success-700"
              : "border-critical-500 bg-critical-50/70 text-critical-700"
          }`}
        >
          {toast.text}
        </div>
      )}

      <div className="mt-10">
        <SectionHeader number={1} title="Загрузка нового файла" />
        <FileDropZone
          onUpload={(file) => uploadMutation.mutate(file)}
          uploading={uploadMutation.isPending}
          uploadPercent={uploadPercent}
        />
      </div>

      <div ref={previewRef} className="mt-12">
        {current && (
          <>
            <SectionHeader number={2} title="Превью датасета" />
            <DatasetPreview
              data={current}
              onDelete={() =>
                onDeleteRequest(current.id, current.original_filename)
              }
              onAnalyze={() => {
                setAnalyzeError(null);
                setAnalyzeOpen(true);
              }}
            />
          </>
        )}
      </div>

      <StartAnalysisModal
        open={analyzeOpen}
        columns={current?.preview.columns ?? []}
        isPending={startAnalysisMutation.isPending}
        errorText={analyzeError}
        onClose={() => setAnalyzeOpen(false)}
        onSubmit={(params) => startAnalysisMutation.mutate(params)}
      />

      <DeleteDatasetModal
        open={deleteTarget !== null}
        datasetId={deleteTarget?.id ?? null}
        filename={deleteTarget?.filename ?? null}
        onClose={() => setDeleteTarget(null)}
        onDeleted={onDeleted}
      />

      <section className="mt-12">
        <SectionHeader
          number={current ? 3 : 2}
          title="Мои датасеты"
          note={
            listQuery.data
              ? `${listQuery.data.length} ${pluralize(
                  listQuery.data.length,
                  PLURAL_FORMS.record,
                )}`
              : undefined
          }
        />
        {listQuery.isPending ? (
          <div className="flex items-center gap-2 border-l-[3px] border-info-500 bg-paper-50 px-4 py-3 font-sans text-sm text-paper-600">
            <Loader2 className="h-4 w-4 animate-spin text-info-500" />
            Загрузка списка…
          </div>
        ) : listQuery.isError ? (
          <p className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
            Не удалось загрузить список датасетов.
          </p>
        ) : listQuery.data && listQuery.data.length > 0 ? (
          <ol className="divide-y divide-paper-200 border-y border-paper-200">
            {listQuery.data.map((d, idx) => (
              <DatasetRow
                key={d.id}
                index={idx + 1}
                dataset={d}
                isActive={current?.id === d.id}
                isOpening={
                  openMutation.isPending && openMutation.variables === d.id
                }
                isDeleting={
                  deleteTarget !== null && deleteTarget.id === d.id
                }
                onOpen={() => openMutation.mutate(d.id)}
                onDelete={() => onDeleteRequest(d.id, d.original_filename)}
              />
            ))}
          </ol>
        ) : (
          <p className="border-l-[3px] border-paper-400 bg-paper-100/60 px-4 py-2 font-serif text-sm text-paper-600">
            У вас пока нет загруженных датасетов.
          </p>
        )}
      </section>
    </div>
  );
}

function SectionHeader({
  number,
  title,
  note,
}: {
  number: number;
  title: string;
  note?: string;
}) {
  return (
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
  );
}

function DatasetRow({
  index,
  dataset,
  isActive,
  isOpening,
  isDeleting,
  onOpen,
  onDelete,
}: {
  index: number;
  dataset: Dataset;
  isActive: boolean;
  isOpening: boolean;
  isDeleting: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const sizeRows =
    dataset.n_rows !== null && dataset.n_cols !== null
      ? `${formatNumber(dataset.n_rows)} строк × ${formatNumber(dataset.n_cols)} колонок`
      : null;

  return (
    <li
      className={`grid grid-cols-[2.5rem_1fr_auto] items-start gap-4 px-1 py-4 ${
        isActive ? "border-l-[3px] border-ink-700 bg-paper-100/50 pl-3" : ""
      }`}
    >
      <span className="pt-0.5 font-mono text-sm text-paper-400">
        {String(index).padStart(2, "0")}.
      </span>
      <div className="min-w-0">
        <p className="truncate font-serif text-[1.0625rem] font-semibold leading-snug text-paper-900">
          {dataset.original_filename}
        </p>
        <p className="mt-1 font-mono text-xs text-paper-500">
          {sizeRows && <>{sizeRows} · </>}
          {formatBytes(dataset.file_size_bytes)} · {formatDateTime(dataset.uploaded_at)}
        </p>
        <p className="mt-1 font-sans text-[0.6875rem] uppercase tracking-wider text-paper-500">
          ФОРМАТ:{" "}
          <span className="font-mono normal-case tracking-normal text-paper-700">
            {dataset.format}
          </span>
        </p>
      </div>
      <div className="flex shrink-0 gap-2">
        <button
          type="button"
          onClick={onOpen}
          disabled={isOpening}
          className="inline-flex items-center gap-1.5 border border-ink-700 bg-paper-50 px-3 py-1.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700 transition-colors hover:bg-ink-700 hover:text-paper-50 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isOpening && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          ОТКРЫТЬ
        </button>
        <button
          type="button"
          onClick={onDelete}
          disabled={isDeleting}
          className="inline-flex items-center gap-1.5 border border-paper-400 bg-paper-50 px-3 py-1.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-critical-500 hover:text-critical-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isDeleting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          УДАЛИТЬ
        </button>
      </div>
    </li>
  );
}

function extractAnalysisError(err: unknown): string {
  const e = err as {
    response?: { status?: number; data?: { detail?: unknown } };
  };
  const detail = e.response?.data?.detail;
  if (e.response?.status === 400 && typeof detail === "string") return detail;
  if (e.response?.status === 404) return "Датасет не найден.";
  if (typeof detail === "string") return detail;
  return "Не удалось запустить анализ. Попробуйте ещё раз.";
}

function extractDatasetError(err: unknown): string {
  const e = err as {
    response?: { status?: number; data?: { detail?: unknown } };
    message?: string;
  };
  const status = e.response?.status;
  const detail = e.response?.data?.detail;

  if (status === 413) return `Файл превышает 100 МБ.`;
  if (status === 415) return "Поддерживаются только файлы CSV и XLSX.";
  if (status === 400) {
    return typeof detail === "string"
      ? detail
      : "Не удалось распарсить файл. Проверьте формат и кодировку.";
  }
  if (!e.response) {
    return "Не удалось связаться с сервером, попробуйте позже.";
  }
  return typeof detail === "string"
    ? detail
    : "Не удалось загрузить файл. Попробуйте ещё раз.";
}
