// TypeScript-зеркала Pydantic-схем из backend/app/schemas/analysis.py.

export type AnalysisStatus = "pending" | "running" | "done" | "failed";
export type Severity = "info" | "warning" | "critical";

export type Analysis = {
  id: string;
  dataset_id: string;
  status: AnalysisStatus;
  target_column: string | null;
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
};

// Облегчённая запись для страницы /history (см. AnalysisListItem на бэке).
// Не путать с Analysis — там более полная модель для polling одного анализа.
export type AnalysisListItem = {
  id: string;
  dataset_id: string;
  dataset_name: string;
  status: AnalysisStatus;
  target_column: string | null;
  recommended_task_type: string | null;
  started_at: string;
  finished_at: string | null;
};

export type AnalysisListResponse = {
  items: AnalysisListItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
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
  task_recommendation: TaskRecommendation | null;
  embedding: number[] | null;
};

// Зеркала Pydantic-схем из backend/app/schemas/task_recommendation.py
// и backend/app/schemas/baseline.py.

export type TaskTypeCode =
  | "BINARY_CLASSIFICATION"
  | "MULTICLASS_CLASSIFICATION"
  | "REGRESSION"
  | "CLUSTERING"
  | "NOT_READY";

export type RecommendationSource = "rules" | "ml" | "hybrid";

export type AppliedRule = {
  code: string;
  description: string;
};

export type TaskRecommendation = {
  task_type_code: TaskTypeCode;
  confidence: number;
  source: RecommendationSource;
  applied_rules: AppliedRule[];
  ml_probabilities: Record<string, number> | null;
  explanation: string;
};

// `mean ± std` пара по фолдам кросс-валидации.
export type MetricValue = {
  mean: number;
  std: number;
};

export type BaselineResult = {
  models: string[];
  metrics: Record<string, Record<string, MetricValue>>;
  feature_importance: Record<string, number>;
  excluded_columns_due_to_leakage: string[];
  n_rows_used: number;
  n_features_used: number;
  trained_at: string;
  // Для CLUSTERING / NOT_READY backend кладёт текстовое объяснение
  // вместо обученных моделей.
  note?: string;
};

export type BaselineStatus = "not_started" | "running" | "done" | "failed";

export type BaselineResponse = {
  baseline_status: BaselineStatus;
  baseline: BaselineResult | null;
  baseline_error: string | null;
};

export type BaselineStartResponse = {
  analysis_id: string;
  baseline_status: "running" | "done";
};

export type SimilarDataset = {
  id: string;
  title: string;
  description: string | null;
  source: string;
  source_url: string | null;
  task_type_code: string;
  distance: number;
};

export type StartAnalysisRequest = {
  target_column?: string | null;
};
