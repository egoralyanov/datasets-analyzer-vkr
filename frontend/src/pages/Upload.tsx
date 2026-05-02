import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileSpreadsheet, FileText, Loader2 } from "lucide-react";
import { datasetsApi } from "../api/datasets";
import type { Dataset, DatasetWithPreview } from "../types/dataset";
import { formatBytes, formatDateTime, formatNumber } from "../lib/format";
import { FileDropZone } from "../components/upload/FileDropZone";
import { DatasetPreview } from "../components/upload/DatasetPreview";

type Toast = { kind: "success" | "error"; text: string } | null;

export function Upload() {
  const queryClient = useQueryClient();
  const previewRef = useRef<HTMLDivElement>(null);

  const [current, setCurrent] = useState<DatasetWithPreview | null>(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [toast, setToast] = useState<Toast>(null);

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

  const deleteMutation = useMutation({
    mutationFn: (id: string) => datasetsApi.remove(id),
    onSuccess: (_data, id) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      if (current?.id === id) setCurrent(null);
      setToast({ kind: "success", text: "Датасет удалён." });
    },
    onError: () => {
      setToast({ kind: "error", text: "Не удалось удалить датасет." });
    },
  });

  const onDelete = (id: string, name: string) => {
    if (!window.confirm(`Удалить датасет «${name}»?`)) return;
    deleteMutation.mutate(id);
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Загрузка датасета</h1>
      <p className="mt-1 text-sm text-slate-600">
        Поддерживаются CSV и XLSX. Файл сохраняется только для вашего аккаунта.
      </p>

      {toast && (
        <div
          className={`mt-4 rounded-md p-3 text-sm border ${
            toast.kind === "success"
              ? "border-green-200 bg-green-50 text-green-800"
              : "border-red-200 bg-red-50 text-red-800"
          }`}
        >
          {toast.text}
        </div>
      )}

      <div className="mt-6">
        <FileDropZone
          onUpload={(file) => uploadMutation.mutate(file)}
          uploading={uploadMutation.isPending}
          uploadPercent={uploadPercent}
        />
      </div>

      <div ref={previewRef} className="mt-8">
        {current && (
          <DatasetPreview
            data={current}
            onDelete={() => onDelete(current.id, current.original_filename)}
          />
        )}
      </div>

      <section className="mt-10">
        <h2 className="text-lg font-semibold text-slate-900">Мои датасеты</h2>
        {listQuery.isPending ? (
          <div className="mt-4 flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            Загрузка…
          </div>
        ) : listQuery.isError ? (
          <p className="mt-4 text-sm text-red-700">
            Не удалось загрузить список датасетов.
          </p>
        ) : listQuery.data && listQuery.data.length > 0 ? (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {listQuery.data.map((d) => (
              <DatasetCard
                key={d.id}
                dataset={d}
                isActive={current?.id === d.id}
                isOpening={openMutation.isPending && openMutation.variables === d.id}
                isDeleting={deleteMutation.isPending && deleteMutation.variables === d.id}
                onOpen={() => openMutation.mutate(d.id)}
                onDelete={() => onDelete(d.id, d.original_filename)}
              />
            ))}
          </div>
        ) : (
          <p className="mt-4 text-sm text-slate-500">
            У вас пока нет загруженных датасетов.
          </p>
        )}
      </section>
    </div>
  );
}

function DatasetCard({
  dataset,
  isActive,
  isOpening,
  isDeleting,
  onOpen,
  onDelete,
}: {
  dataset: Dataset;
  isActive: boolean;
  isOpening: boolean;
  isDeleting: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const Icon = dataset.format === "xlsx" ? FileSpreadsheet : FileText;
  const iconColor = dataset.format === "xlsx" ? "text-emerald-600" : "text-blue-600";
  const ringClass = isActive ? "ring-2 ring-blue-500" : "hover:shadow-sm";

  return (
    <div
      className={`rounded-lg border border-slate-200 bg-white p-4 transition-shadow ${ringClass}`}
    >
      <div className="flex items-start gap-3">
        <Icon className={`h-5 w-5 shrink-0 ${iconColor}`} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-slate-900">
            {dataset.original_filename}
          </p>
          <p className="mt-0.5 text-xs text-slate-500">
            {dataset.n_rows !== null && dataset.n_cols !== null
              ? `${formatNumber(dataset.n_rows)} строк × ${formatNumber(
                  dataset.n_cols,
                )} колонок · `
              : ""}
            {formatBytes(dataset.file_size_bytes)} ·{" "}
            {formatDateTime(dataset.uploaded_at)}
          </p>
        </div>
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button
          type="button"
          onClick={onOpen}
          disabled={isOpening}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-900 hover:bg-slate-50 disabled:opacity-60"
        >
          {isOpening && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          Открыть
        </button>
        <button
          type="button"
          onClick={onDelete}
          disabled={isDeleting}
          className="inline-flex items-center gap-1.5 rounded-md border border-red-200 bg-white px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-60"
        >
          {isDeleting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          Удалить
        </button>
      </div>
    </div>
  );
}

// Карта серверных кодов в человеческие сообщения для загрузки датасета.
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
