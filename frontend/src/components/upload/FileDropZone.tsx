// Drop-zone выбора и загрузки датасета.
//
// Стиль (Sprint 5, Phase 3.5): scientific/archive — внешняя hairline-обводка
// с декоративной двойной рамкой и `+`-метками в углах. Когда файл выбран —
// dropzone сворачивается в одну компактную ruled-строку «имя файла · размер
// · [Убрать] [Загрузить]». Большая инструкция «перетащите...» уходит, чтобы
// не отвлекать от уже сделанного выбора.
import { useRef, useState } from "react";
import { Loader2 } from "lucide-react";
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
    // Сбрасываем input, чтобы можно было выбрать тот же файл повторно после «Убрать».
    e.target.value = "";
  };
  const onClear = () => {
    setFile(null);
    setError(null);
  };
  const onSubmit = () => {
    if (file) onUpload(file);
  };

  // Компактный режим: файл выбран, ruled-строка с действиями.
  if (file) {
    return (
      <div className="space-y-3">
        <div
          className={`flex flex-wrap items-center justify-between gap-4 border-l-[3px] px-5 py-3 ${
            uploading
              ? "border-info-500 bg-info-50/40"
              : "border-ink-700 bg-paper-50"
          }`}
        >
          <div className="min-w-0 flex-1">
            <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
              ФАЙЛ ВЫБРАН
            </p>
            <p className="mt-1 truncate font-mono text-sm text-paper-800">
              {file.name}
            </p>
            <p className="mt-0.5 font-mono text-xs text-paper-500">
              {formatBytes(file.size)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClear}
              disabled={uploading}
              className="border border-paper-400 bg-paper-50 px-3 py-1.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              ✕ УБРАТЬ
            </button>
            <button
              type="button"
              onClick={onSubmit}
              disabled={uploading}
              className="inline-flex items-center gap-2 border border-ink-700 bg-ink-700 px-4 py-1.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {uploading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              ЗАГРУЗИТЬ
            </button>
          </div>
        </div>
        {uploading && (
          <div>
            <div className="h-1 w-full bg-paper-200">
              <div
                className="h-1 bg-ink-700 transition-all"
                style={{ width: `${uploadPercent}%` }}
              />
            </div>
            <p className="mt-1 font-mono text-xs text-paper-500">
              ЗАГРУЗКА · {uploadPercent}%
            </p>
          </div>
        )}
      </div>
    );
  }

  // Полный режим: ничего не выбрано — большая «архивная карточка».
  return (
    <div className="space-y-3">
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onPick}
        className={`relative cursor-pointer border bg-paper-50 px-8 py-12 text-center transition-colors ${
          dragActive
            ? "border-ink-700 bg-ink-700 text-paper-50"
            : "border-paper-300 hover:border-ink-700"
        }`}
        style={{
          // Декоративная внутренняя двойная рамка через inset-shadow
          // (см. DESIGN_TOKENS.md, раздел 8.3).
          boxShadow: dragActive
            ? "inset 0 0 0 1px #FAF8F4"
            : "inset 0 0 0 1px #D2CCBE",
          backgroundClip: "padding-box",
        }}
      >
        <CornerMark inverted={dragActive} className="absolute left-3 top-3" />
        <CornerMark inverted={dragActive} className="absolute right-3 top-3" />
        <CornerMark inverted={dragActive} className="absolute left-3 bottom-3" />
        <CornerMark
          inverted={dragActive}
          className="absolute right-3 bottom-3"
        />

        <p
          className={`font-serif text-[1.5rem] font-semibold leading-snug tracking-tight ${
            dragActive ? "text-paper-50" : "text-paper-900"
          }`}
        >
          {dragActive ? "Отпустите файл здесь" : "Перетащите файл"}
        </p>
        <p
          className={`mt-1 font-sans text-sm ${
            dragActive ? "text-paper-100" : "text-paper-500"
          }`}
        >
          {dragActive ? "" : "или нажмите, чтобы выбрать"}
        </p>
        <p
          className={`mt-3 font-mono text-xs ${
            dragActive ? "text-paper-200" : "text-paper-500"
          }`}
        >
          CSV / XLSX · до {MAX_FILE_SIZE_MB} МБ
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx"
        className="hidden"
        onChange={onInputChange}
      />

      {error && (
        <div className="border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
          {error}
        </div>
      )}
    </div>
  );
}

function CornerMark({
  inverted,
  className = "",
}: {
  inverted: boolean;
  className?: string;
}) {
  return (
    <span
      className={`font-mono text-xs ${
        inverted ? "text-paper-200" : "text-paper-400"
      } ${className}`}
      aria-hidden="true"
    >
      +
    </span>
  );
}
