// Графики профайлинга: гистограммы числовых, bar chart категориальных,
// heatmap корреляций. Plotly через react-plotly.js — интерактивный (zoom, hover).
//
// Default-импорт `import Plot from "react-plotly.js"` несовместим с React 19
// (factory выдаёт legacy-class, обёрнутый Babel в объект, а не функцию-компонент).
// Создаём компонент явно через factory + plotly.js — это надёжный путь,
// рекомендованный в .knowledge/troubleshooting.md.
// react-plotly.js и plotly.js-dist-min — CommonJS-зависимости.
// Pre-bundling в Vite (см. optimizeDeps в vite.config.ts) гарантирует,
// что default-импорт даёт саму функцию-компонент, а не CJS-обёртку.
import Plot from "react-plotly.js";
import { BarChart3 } from "lucide-react";
import type { MetaFeatures } from "../../types/analysis";

type Props = {
  meta: MetaFeatures;
};

const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

const PLOT_LAYOUT_BASE = {
  autosize: true,
  margin: { t: 30, r: 20, b: 50, l: 50 },
  font: { family: "system-ui, -apple-system, sans-serif", size: 11 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
};

const NUMERIC_COLORS = "#2563eb"; // blue-600
const CATEGORICAL_COLORS = "#10b981"; // emerald-500

export function Distributions({ meta }: Props) {
  const numeric = meta.distributions?.numeric ?? {};
  const categorical = meta.distributions?.categorical ?? {};
  const correlationMatrix = meta.correlation_matrix;
  const targetCounts = meta.target_value_counts;

  const numericEntries = Object.entries(numeric).slice(0, 5);
  const categoricalEntries = Object.entries(categorical).slice(0, 4);
  const hasAny =
    numericEntries.length > 0 ||
    categoricalEntries.length > 0 ||
    !!correlationMatrix ||
    !!targetCounts;

  if (!hasAny) {
    return null;
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6">
      <div className="flex items-center gap-2">
        <BarChart3 className="h-5 w-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-slate-900">Распределения</h2>
      </div>

      {targetCounts && (
        <div className="mt-4">
          <h3 className="text-sm font-medium text-slate-700">
            Распределение целевой переменной
          </h3>
          <Plot
            data={[
              {
                type: "bar",
                x: Object.keys(targetCounts),
                y: Object.values(targetCounts),
                marker: { color: "#7c3aed" }, // violet-600
              },
            ]}
            layout={{
              ...PLOT_LAYOUT_BASE,
              height: 260,
              xaxis: { title: { text: "Класс" }, type: "category" },
              yaxis: { title: { text: "Количество" } },
            }}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </div>
      )}

      {numericEntries.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-slate-700">
            Гистограммы числовых признаков (первые {numericEntries.length})
          </h3>
          <div className="mt-2 grid gap-4 lg:grid-cols-2">
            {numericEntries.map(([col, dist]) => {
              const edges = dist.bin_edges;
              // Центры бинов как метки x: середина между соседними edges.
              const centers = edges
                .slice(0, -1)
                .map((e, i) => (e + edges[i + 1]) / 2);
              return (
                <div
                  key={col}
                  className="rounded border border-slate-200 p-3"
                >
                  <p className="text-xs font-medium text-slate-700">{col}</p>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        x: centers,
                        y: dist.counts,
                        marker: { color: NUMERIC_COLORS },
                      },
                    ]}
                    layout={{
                      ...PLOT_LAYOUT_BASE,
                      height: 200,
                      bargap: 0.05,
                      xaxis: { title: { text: "Значение" } },
                      yaxis: { title: { text: "Частота" } },
                    }}
                    config={PLOT_CONFIG}
                    style={{ width: "100%" }}
                    useResizeHandler
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {categoricalEntries.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-slate-700">
            Распределения категориальных признаков (топ-{categoricalEntries.length})
          </h3>
          <div className="mt-2 grid gap-4 lg:grid-cols-2">
            {categoricalEntries.map(([col, dist]) => (
              <div
                key={col}
                className="rounded border border-slate-200 p-3"
              >
                <p className="text-xs font-medium text-slate-700">
                  {col}
                  {dist.other_count > 0 && (
                    <span className="ml-2 text-slate-500">
                      (+{dist.other_count} прочих)
                    </span>
                  )}
                </p>
                <Plot
                  data={[
                    {
                      type: "bar",
                      x: dist.categories,
                      y: dist.counts,
                      marker: { color: CATEGORICAL_COLORS },
                    },
                  ]}
                  layout={{
                    ...PLOT_LAYOUT_BASE,
                    height: 220,
                    xaxis: { title: { text: "Категория" }, type: "category" },
                    yaxis: { title: { text: "Количество" } },
                  }}
                  config={PLOT_CONFIG}
                  style={{ width: "100%" }}
                  useResizeHandler
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {correlationMatrix && Object.keys(correlationMatrix).length >= 2 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-slate-700">
            Матрица корреляций (Пирсон)
          </h3>
          <CorrelationHeatmap matrix={correlationMatrix} />
        </div>
      )}
    </section>
  );
}

function CorrelationHeatmap({
  matrix,
}: {
  matrix: Record<string, Record<string, number>>;
}) {
  const labels = Object.keys(matrix);
  const z = labels.map((row) => labels.map((col) => matrix[row][col] ?? 0));

  return (
    <Plot
      data={[
        {
          type: "heatmap",
          x: labels,
          y: labels,
          z,
          zmin: -1,
          zmax: 1,
          colorscale: "RdBu",
          reversescale: true,
          hoverongaps: false,
          colorbar: { title: { text: "r" }, thickness: 12 },
        },
      ]}
      layout={{
        ...PLOT_LAYOUT_BASE,
        height: Math.max(280, 60 + labels.length * 28),
        xaxis: { tickangle: -30 },
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
