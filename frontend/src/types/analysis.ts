// TypeScript-зеркала Pydantic-схем из backend/app/schemas/analysis.py.

export type AnalysisStatus = "pending" | "running" | "done" | "failed";
export type Severity = "info" | "warning" | "critical";

export type Analysis = {
  id: string;
  dataset_id: string;
  status: AnalysisStatus;
  target_column: string | null;
  hinted_task_type: string | null;
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
};

export type QualityFlag = {
  rule_code: string;
  severity: Severity;
  rule_name: string;
  message: string;
  context: Record<string, unknown> | null;
};

// MetaFeatures — гибкая запись: 33+ ключей в JSONB, не хочется типизировать
// каждое поле. Известные «горячие» ключи объявлены явно для удобства,
// остальное — через индекс-сигнатуру.
export type MetaFeatures = {
  n_rows: number;
  n_cols: number;
  dtype_counts: Record<string, number>;
  memory_mb: number;
  total_missing_pct: number;
  max_col_missing_pct: number;
  duplicate_rows_pct: number;
  target_kind?: "categorical" | "regression" | null;
  target_imbalance_ratio?: number | null;
  target_class_entropy?: number | null;
  target_value_counts?: Record<string, number> | null;
  target_correlation_max?: number | null;
  target_mutual_information_max?: number | null;
  max_abs_correlation?: number | null;
  correlation_matrix?: Record<string, Record<string, number>>;
  distributions?: {
    numeric?: Record<string, { bin_edges: number[]; counts: number[] }>;
    categorical?: Record<
      string,
      { categories: string[]; counts: number[]; other_count: number }
    >;
  };
  sampling?: {
    sampled: boolean;
    sample_size: number;
    original_size: number;
  };
  missing_by_column?: Record<string, number>;
  high_cardinality_cols?: string[];
  low_variance_numeric_cols?: string[];
  low_variance_categorical_cols?: string[];
  outliers_by_column?: Record<string, number>;
  // Прочие поля meta_features.
  [key: string]: unknown;
};

export type AnalysisResult = {
  analysis: Analysis;
  meta_features: MetaFeatures;
  flags: QualityFlag[];
};

export type StartAnalysisRequest = {
  target_column?: string | null;
  hinted_task_type?: string | null;
};

export type HintedTaskType =
  | "binary_classification"
  | "multiclass_classification"
  | "regression"
  | "clustering";
