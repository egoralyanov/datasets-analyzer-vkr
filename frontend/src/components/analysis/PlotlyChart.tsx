// Минимальная React-обёртка над plotly.js без зависимости от react-plotly.js.
//
// react-plotly.js@2.6.0 несовместим с React 19: и default-импорт,
// и factory-фабрика выдают объект вместо функции-компонента (CJS interop +
// legacy-class через Babel), что приводит к React error #130 на рендере.
// Здесь работаем с нативным API plotly.js напрямую — Plotly.newPlot/purge
// в useEffect. Этот путь стабилен и не зависит от особенностей бандлера.
import { useEffect, useRef } from "react";
// @ts-expect-error — plotly.js-dist-min поставляется без типов.
import Plotly from "plotly.js-dist-min";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotData = any[];
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotLayout = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PlotConfig = any;

interface PlotlyChartProps {
  data: PlotData;
  layout?: PlotLayout;
  config?: PlotConfig;
  style?: React.CSSProperties;
  className?: string;
}

const DEFAULT_CONFIG: PlotConfig = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false,
};

export function PlotlyChart({
  data,
  layout,
  config,
  style,
  className,
}: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    Plotly.newPlot(node, data, layout ?? {}, { ...DEFAULT_CONFIG, ...(config ?? {}) });
    return () => {
      Plotly.purge(node);
    };
  }, [data, layout, config]);

  return <div ref={ref} style={style} className={className} />;
}
