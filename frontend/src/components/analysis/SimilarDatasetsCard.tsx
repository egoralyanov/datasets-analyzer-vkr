// Top-K похожих датасетов из каталога. Архивная подача: каждая строка —
// «карточка библиотечного каталога» с кодом записи в Mono, hairline-руллями
// между записями, без скруглений и теней.
//
// См. frontend/DESIGN_TOKENS.md, раздел 8.4 + manifesto.
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { analysesApi } from "../../api/analyses";
import type { SimilarDataset } from "../../types/analysis";

type Props = {
  analysisId: string;
};

const DESCRIPTION_TRIM = 200;

function trimDescription(text: string | null | undefined): {
  short: string;
  hasMore: boolean;
} {
  if (!text) return { short: "", hasMore: false };
  if (text.length <= DESCRIPTION_TRIM) return { short: text, hasMore: false };
  return {
    short: `${text.slice(0, DESCRIPTION_TRIM).trimEnd()}…`,
    hasMore: true,
  };
}

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
  const { short: shortDesc, hasMore } = trimDescription(item.description);

  return (
    <li className="grid grid-cols-[2.5rem_1fr] gap-4 px-1 py-4">
      <span className="font-mono text-sm text-paper-400">
        {String(index).padStart(2, "0")}.
      </span>
      <div>
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h3 className="font-serif text-[1.125rem] font-semibold leading-snug text-paper-900">
            {item.title}
          </h3>
          <div className="flex items-baseline gap-3 font-mono text-xs text-paper-500">
            <span className="text-paper-700">{item.source}</span>
            <span>
              cos d{" "}
              <span className="text-paper-800">{item.distance.toFixed(3)}</span>
            </span>
          </div>
        </div>

        {shortDesc && (
          <p
            className="mt-2 font-serif text-sm leading-relaxed text-paper-600"
            title={hasMore ? item.description ?? undefined : undefined}
          >
            {shortDesc}
          </p>
        )}

        <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 font-sans text-xs text-paper-500">
          <span className="uppercase tracking-wider">
            ТИП:{" "}
            <span className="font-mono normal-case tracking-normal text-paper-700">
              {item.task_type_code.toLowerCase()}
            </span>
          </span>
          {item.source_url && (
            <a
              href={item.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium uppercase tracking-wider text-ink-700 underline-offset-2 hover:underline"
            >
              ОТКРЫТЬ ↗
            </a>
          )}
        </div>
      </div>
    </li>
  );
}
