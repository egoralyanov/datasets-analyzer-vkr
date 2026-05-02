import { useRef, useState } from "react";
import { Loader2, Upload as UploadIcon, X } from "lucide-react";
import {
  ALLOWED_DATASET_EXTENSIONS,
  MAX_FILE_SIZE_BYTES,
  MAX_FILE_SIZE_MB,
} from "../../lib/constants";
import { formatBytes } from "../../lib/format";

interface Props {
  onUpload: (file: File) => void;
  uploading: boolean;
  uploadPercent: number;
}

function getExtension(filename: string): string | null {
  if (!filename || !filename.includes(".")) return null;
  return filename.split(".").pop()!.toLowerCase();
}

function validateFile(file: File): string | null {
  const ext = getExtension(file.name);
  if (!ext || !ALLOWED_DATASET_EXTENSIONS.includes(ext as "csv" | "xlsx")) {
    return "Поддерживаются только файлы CSV и XLSX.";
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `Файл превышает лимит ${MAX_FILE_SIZE_MB} МБ.`;
  }
  return null;
}

export function FileDropZone({ onUpload, uploading, uploadPercent }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const acceptFile = (f: File) => {
    const err = validateFile(f);
    if (err) {
      setError(err);
      setFile(null);
      return;
    }
    setError(null);
    setFile(f);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };
  const onDragLeave = () => setDragActive(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) acceptFile(f);
  };
  const onPick = () => inputRef.current?.click();
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) acceptFile(f);
    // Сбрасываем input, чтобы можно было выбрать тот же файл повторно после "Убрать".
    e.target.value = "";
  };
  const onClear = () => {
    setFile(null);
    setError(null);
  };
  const onSubmit = () => {
    if (file) onUpload(file);
  };

  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
        dragActive ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-white"
      }`}
    >
      <UploadIcon className="mx-auto h-8 w-8 text-slate-400" />
      <p className="mt-3 text-sm text-slate-700">
        {dragActive
          ? "Отпустите файл здесь"
          : "Перетащите CSV или XLSX-файл сюда"}
      </p>
      <p className="mt-1 text-xs text-slate-500">
        или выберите файл вручную (до {MAX_FILE_SIZE_MB} МБ)
      </p>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        className="hidden"
        onChange={onInputChange}
      />

      {!file && !error && (
        <button
          type="button"
          onClick={onPick}
          className="mt-4 inline-flex items-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          Выбрать файл
        </button>
      )}

      {error && (
        <div className="mt-4 mx-auto max-w-md rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {file && (
        <div className="mt-4 mx-auto max-w-md rounded-md border border-slate-200 bg-slate-50 p-3 text-left">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-900">
                {file.name}
              </p>
              <p className="text-xs text-slate-500">{formatBytes(file.size)}</p>
            </div>
            <button
              type="button"
              onClick={onClear}
              disabled={uploading}
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-slate-600 hover:bg-slate-200 disabled:opacity-50"
              aria-label="Убрать файл"
            >
              <X className="h-3.5 w-3.5" />
              Убрать
            </button>
          </div>

          {uploading && (
            <div className="mt-3">
              <div className="h-2 w-full rounded-full bg-slate-200">
                <div
                  className="h-2 rounded-full bg-blue-600 transition-all"
                  style={{ width: `${uploadPercent}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Загрузка… {uploadPercent}%
              </p>
            </div>
          )}

          <button
            type="button"
            onClick={onSubmit}
            disabled={uploading}
            className="mt-3 inline-flex items-center justify-center gap-2 w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {uploading && <Loader2 className="h-4 w-4 animate-spin" />}
            Загрузить
          </button>
        </div>
      )}
    </div>
  );
}
