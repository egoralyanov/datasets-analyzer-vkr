// Графики профайлинга: гистограммы числовых, bar chart категориальных,
// heatmap корреляций. Используется собственная обёртка над plotly.js —
// см. ./PlotlyChart.tsx, там же объяснение почему отказались от
// react-plotly.js (несовместимость с React 19).
//
// Стиль (Sprint 5, Phase 2): контейнер — hairline-карточка без скругления,
// заголовок графика как «Рис. N. ...» в Plex Sans tracking-wider. Цвета
// согласованы с DESIGN_TOKENS.md, раздел 1.4: numeric=ink.700,
// categorical=success.500, target=critical.500, heatmap diverging
// brick→paper→ink.
import { PlotlyChart } from "./PlotlyChart";
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
  margin: { t: 20, r: 20, b: 50, l: 55 },
  font: {
    family: '"IBM Plex Sans", "Source Sans 3", system-ui, sans-serif',
    size: 11,
    color: "#4F4A40", // paper.600
  },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: {
    gridcolor: "#E5E0D5", // paper.200
    linecolor: "#A89F8E", // paper.400
    zerolinecolor: "#D2CCBE", // paper.300
  },
  yaxis: {
    gridcolor: "#E5E0D5",
    linecolor: "#A89F8E",
    zerolinecolor: "#D2CCBE",
  },
};

const COLOR_NUMERIC = "#1F2A44"; // ink.700
const COLOR_CATEGORICAL = "#5A7A55"; // success.500
const COLOR_TARGET = "#A53A2A"; // critical.500

// Diverging-палитра brick → paper → ink. Используется для heatmap корреляций
// взамен RdBu_r — попадает в основную палитру и не вносит «постороннего» синего.
const CORRELATION_COLORSCALE: [number, string][] = [
  [0.0, "#A53A2A"], // critical.500
  [0.25, "#D8A89A"],
  [0.5, "#FAF8F4"], // paper.50
  [0.75, "#7D8AA6"],
  [1.0, "#1F2A44"], // ink.700
];

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

  let figureCounter = 0;
  const nextFigure = () => ++figureCounter;

  return (
    <div className="space-y-8">
      {targetCounts && (
        <Figure
          number={nextFigure()}
          caption={`Распределение целевой переменной (${
            Object.keys(targetCounts).length
          } класса)`}
        >
          <PlotlyChart
            data={[
              {
                type: "bar",
                x: Object.keys(targetCounts),
                y: Object.values(targetCounts),
                marker: { color: COLOR_TARGET },
              },
            ]}
            layout={{
              ...PLOT_LAYOUT_BASE,
              height: 260,
              xaxis: {
                ...PLOT_LAYOUT_BASE.xaxis,
                title: { text: "Класс" },
                type: "category",
              },
              yaxis: {
                ...PLOT_LAYOUT_BASE.yaxis,
                title: { text: "Количество" },
              },
            }}
            config={PLOT_CONFIG}
            style={{ width: "100%" }}
          />
        </Figure>
      )}

      {numericEntries.length > 0 && (
        <div>
          <SubsectionLabel>
            Гистограммы числовых признаков (первые {numericEntries.length})
          </SubsectionLabel>
          <div className="mt-3 grid gap-6 lg:grid-cols-2">
            {numericEntries.map(([col, dist]) => {
              const edges = dist.bin_edges;
              // Центры бинов как метки x: середина между соседними edges.
              const centers = edges
                .slice(0, -1)
                .map((e, i) => (e + edges[i + 1]) / 2);
              const n = dist.counts.reduce((a, b) => a + b, 0);
              return (
                <Figure
                  key={col}
                  number={nextFigure()}
                  caption={
                    <>
                      Распределение признака{" "}
                      <span className="font-mono">{col}</span> (n=
                      <span className="font-mono">{n}</span>, bins=
                      <span className="font-mono">{dist.counts.length}</span>)
                    </>
                  }
                >
                  <PlotlyChart
                    data={[
                      {
                        type: "bar",
                        x: centers,
                        y: dist.counts,
                        marker: { color: COLOR_NUMERIC },
                      },
                    ]}
                    layout={{
                      ...PLOT_LAYOUT_BASE,
                      height: 200,
                      bargap: 0.05,
                      xaxis: {
                        ...PLOT_LAYOUT_BASE.xaxis,
                        title: { text: "Значение" },
                      },
                      yaxis: {
                        ...PLOT_LAYOUT_BASE.yaxis,
                        title: { text: "Частота" },
                      },
                    }}
                    config={PLOT_CONFIG}
                    style={{ width: "100%" }}
                  />
                </Figure>
              );
            })}
          </div>
        </div>
      )}

      {categoricalEntries.length > 0 && (
        <div>
          <SubsectionLabel>
            Категориальные признаки (топ-{categoricalEntries.length})
          </SubsectionLabel>
          <div className="mt-3 grid gap-6 lg:grid-cols-2">
            {categoricalEntries.map(([col, dist]) => (
              <Figure
                key={col}
                number={nextFigure()}
                caption={
                  <>
                    Распределение категориального признака{" "}
                    <span className="font-mono">{col}</span>
                    {dist.other_count > 0 && (
                      <>
                        {" "}
                        (свёрнуто прочих:{" "}
                        <span className="font-mono">{dist.other_count}</span>)
                      </>
                    )}
                  </>
                }
              >
                <PlotlyChart
                  data={[
                    {
                      type: "bar",
                      x: dist.categories,
                      y: dist.counts,
                      marker: { color: COLOR_CATEGORICAL },
                    },
                  ]}
                  layout={{
                    ...PLOT_LAYOUT_BASE,
                    height: 220,
                    xaxis: {
                      ...PLOT_LAYOUT_BASE.xaxis,
                      title: { text: "Категория" },
                      type: "category",
                    },
                    yaxis: {
                      ...PLOT_LAYOUT_BASE.yaxis,
                      title: { text: "Количество" },
                    },
                  }}
                  config={PLOT_CONFIG}
                  style={{ width: "100%" }}
                />
              </Figure>
            ))}
          </div>
        </div>
      )}

      {correlationMatrix && Object.keys(correlationMatrix).length >= 2 && (
        <Figure
          number={nextFigure()}
          caption="Матрица корреляций Пирсона между числовыми признаками"
        >
          <CorrelationHeatmap matrix={correlationMatrix} />
        </Figure>
      )}
    </div>
  );
}

function SubsectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
      {children}
    </h3>
  );
}

function Figure({
  number,
  caption,
  children,
}: {
  number: number;
  caption: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <figure className="border border-paper-300 bg-paper-50 p-4">
      <div className="border-l-[3px] border-paper-300 px-3 py-1">
        {children}
      </div>
      <figcaption className="mt-3 border-t border-paper-200 pt-2 font-sans text-xs text-paper-500">
        <span className="font-medium text-paper-700">Рис. {number}.</span>{" "}
        {caption}
      </figcaption>
    </figure>
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
    <PlotlyChart
      data={[
        {
          type: "heatmap",
          x: labels,
          y: labels,
          z,
          zmin: -1,
          zmax: 1,
          colorscale: CORRELATION_COLORSCALE,
          hoverongaps: false,
          colorbar: { title: { text: "r" }, thickness: 12 },
        },
      ]}
      layout={{
        ...PLOT_LAYOUT_BASE,
        height: Math.max(280, 60 + labels.length * 28),
        xaxis: { ...PLOT_LAYOUT_BASE.xaxis, tickangle: -30 },
      }}
      config={PLOT_CONFIG}
      style={{ width: "100%" }}
    />
  );
}
