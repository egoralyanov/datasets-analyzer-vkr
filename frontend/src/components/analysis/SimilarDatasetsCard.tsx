import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Library, Loader2 } from "lucide-react";
import { analysesApi } from "../../api/analyses";
import type { SimilarDataset } from "../../types/analysis";

type Props = {
  analysisId: string;
};

const SOURCE_PILL: Record<string, string> = {
  sklearn: "bg-blue-50 text-blue-700 border-blue-200",
  uci: "bg-emerald-50 text-emerald-700 border-emerald-200",
  github: "bg-slate-100 text-slate-700 border-slate-200",
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
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <Library className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">
          Похожие датасеты
        </h2>
      </div>
      <p className="mt-1 text-sm text-slate-600">
        Top-5 близких записей из каталога по косинусной мере между
        мета-признаками.
      </p>

      {query.isLoading && (
        <div className="mt-4 flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin" />
          Подбор похожих…
        </div>
      )}

      {query.isError && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          Не удалось загрузить похожие датасеты.
        </div>
      )}

      {query.data && query.data.length === 0 && (
        <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
          Не удалось подобрать похожие датасеты — для этого анализа embedding не
          сохранён (возможно, не загружен scaler).
        </div>
      )}

      {query.data && query.data.length > 0 && (
        <ul className="mt-4 space-y-3">
          {query.data.map((item) => (
            <SimilarRow key={item.id} item={item} />
          ))}
        </ul>
      )}
    </section>
  );
}

function SimilarRow({ item }: { item: SimilarDataset }) {
  const sourceClass =
    SOURCE_PILL[item.source.toLowerCase()] ??
    "bg-slate-100 text-slate-700 border-slate-200";
  const { short: shortDesc, hasMore } = trimDescription(item.description);

  return (
    <li className="rounded-md border border-slate-200 bg-white p-4 transition-colors hover:border-slate-300 hover:bg-slate-50">
      <div className="flex flex-wrap items-start gap-2">
        <h3 className="flex-1 text-base font-semibold text-slate-900">
          {item.title}
        </h3>
        <span
          className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${sourceClass}`}
        >
          {item.source}
        </span>
      </div>

      {shortDesc && (
        <p
          className="mt-2 text-sm text-slate-600"
          title={hasMore ? item.description ?? undefined : undefined}
        >
          {shortDesc}
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-3 text-slate-500">
          <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-700">
            {item.task_type_code.toLowerCase().replace(/_/g, " ")}
          </span>
          <span>
            cos d ={" "}
            <span className="font-mono text-slate-700">
              {item.distance.toFixed(3)}
            </span>
          </span>
        </div>

        {item.source_url && (
          <a
            href={item.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
          >
            Открыть
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
    </li>
  );
}
