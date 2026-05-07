// Inline-уведомление о дубликате при загрузке датасета (Sprint 6, Phase 5.3).
//
// Бэк (POST /datasets/upload, Phase 4.1) при попытке загрузить файл с уже
// известным file_hash возвращает 409 + {detail, existing_dataset_id}. Раньше
// detail падал в banner-toast "Dataset with identical content already exists",
// что выглядело как ошибка и не давало пользователю никакого пути выхода.
//
// Этот компонент — interactive notice (info-tone, не error). Рендерится на
// месте превью датасета (см. Upload.tsx, область previewRef) и предлагает:
//   1) ОТКРЫТЬ СУЩЕСТВУЮЩИЙ — пробросить existing_dataset_id в openMutation,
//      превью этого датасета подтянется как обычно.
//   2) ВЫБРАТЬ ДРУГОЙ ФАЙЛ — сбросить state выбранного файла (через
//      key-trick в Upload.tsx) и закрыть notice.
//
// Никакого toast'а параллельно с этим блоком не показываем — он самодостаточен.
import { Info, Loader2 } from "lucide-react";

type Props = {
  /** ID существующего датасета с тем же file_hash. Прокинется в openMutation. */
  existingDatasetId: string;
  /** openMutation.isPending — дизейблит «ОТКРЫТЬ СУЩЕСТВУЮЩИЙ» на время запроса. */
  isOpening: boolean;
  onOpenExisting: (existingDatasetId: string) => void;
  onChooseAnother: () => void;
};

export function DuplicateDatasetNotice({
  existingDatasetId,
  isOpening,
  onOpenExisting,
  onChooseAnother,
}: Props) {
  return (
    <div className="flex gap-4 border border-paper-300 border-l-[4px] border-l-info-500 bg-paper-100 px-5 py-4">
      <Info
        className="mt-0.5 h-5 w-5 shrink-0 text-info-700"
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1">
        <h3 className="font-serif text-[1.125rem] font-semibold leading-snug text-paper-900">
          Этот файл уже загружен
        </h3>
        <p className="mt-1.5 font-sans text-sm leading-relaxed text-paper-700">
          В вашей библиотеке есть датасет с идентичным содержимым. Его можно
          использовать для нового анализа без повторной загрузки.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => onOpenExisting(existingDatasetId)}
            disabled={isOpening}
            className="inline-flex items-center gap-2 border border-ink-700 bg-ink-700 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isOpening && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            ОТКРЫТЬ СУЩЕСТВУЮЩИЙ
          </button>
          <button
            type="button"
            onClick={() => onChooseAnother()}
            disabled={isOpening}
            className="border border-paper-400 bg-paper-50 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            ВЫБРАТЬ ДРУГОЙ ФАЙЛ
          </button>
        </div>
      </div>
    </div>
  );
}
