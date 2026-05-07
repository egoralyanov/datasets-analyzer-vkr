// Top-K похожих датасетов из каталога. Архивная подача — нумерованный
// каталог библиотечных карточек с hairline-руллями.
//
// Sprint 6, Phase 2: компактный режим — одна-две строки на запись.
// Заголовок + meta (источник, cosine distance, тип задачи) inline в одной
// строке, описание сократили до 1 строки (truncate). Кнопка «ОТКРЫТЬ»
// справа inline, не на отдельной строке. Нумерация и hairline сохранены.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4.
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { analysesApi } from "../../api/analyses";
import type { SimilarDataset } from "../../types/analysis";

type Props = {
  analysisId: string;
};

export function SimilarDatasetsCard({ analysisId }: Props) {
  const query = useQuery({
    queryKey: ["similar", analysisId],
    queryFn: () => analysesApi.getSimilar(analysisId, 5),
    enabled: !!analysisId,
  });

  return (
    <div>
      <p className="font-serif text-[0.9375rem] leading-relaxed text-paper-600">
        Top-5 близких записей из каталога. Сходство — косинусная мера в
        пространстве 16+1 мета-признаков, индекс HNSW.
      </p>

      {query.isLoading && (
        <div className="mt-4 flex items-center gap-2 border-l-[3px] border-info-500 bg-paper-50 px-4 py-3 font-sans text-sm text-paper-600">
          <Loader2 className="h-4 w-4 animate-spin text-info-500" />
          Подбор похожих…
        </div>
      )}

      {query.isError && (
        <div className="mt-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-3 font-sans text-sm text-critical-700">
          Не удалось загрузить похожие датасеты.
        </div>
      )}

      {query.data && query.data.length === 0 && (
        <div className="mt-4 border-l-[3px] border-paper-400 bg-paper-100/60 px-4 py-3 font-sans text-sm text-paper-600">
          Не удалось подобрать похожие датасеты — для этого анализа embedding не
          сохранён (возможно, не загружен scaler).
        </div>
      )}

      {query.data && query.data.length > 0 && (
        <ol className="mt-4 divide-y divide-paper-200 border-y border-paper-200">
          {query.data.map((item, idx) => (
            <SimilarRow key={item.id} item={item} index={idx + 1} />
          ))}
        </ol>
      )}
    </div>
  );
}

function SimilarRow({
  item,
  index,
}: {
  item: SimilarDataset;
  index: number;
}) {
  return (
    <li className="grid grid-cols-[2.5rem_1fr_auto] items-center gap-4 px-1 py-2.5">
      <span className="font-mono text-sm text-paper-400">
        {String(index).padStart(2, "0")}.
      </span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
          <h3
            className="truncate font-serif text-[1rem] font-semibold leading-snug text-paper-900"
            title={item.title}
          >
            {item.title}
          </h3>
          <span className="font-mono text-xs text-paper-700">
            {item.source}
          </span>
          <span className="font-mono text-xs text-paper-500">
            cos d{" "}
            <span className="text-paper-800">{item.distance.toFixed(3)}</span>
          </span>
          <span className="font-mono text-xs text-paper-500">
            {item.task_type_code.toLowerCase()}
          </span>
        </div>
        {item.description && (
          <p
            className="mt-0.5 truncate font-serif text-xs text-paper-500"
            title={item.description}
          >
            {item.description}
          </p>
        )}
      </div>
      {item.source_url && (
        <a
          href={item.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="border-b border-transparent font-sans text-xs font-medium uppercase tracking-wider text-ink-700 hover:border-ink-700"
        >
          ОТКРЫТЬ ↗
        </a>
      )}
    </li>
  );
}
