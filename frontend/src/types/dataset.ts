// Зеркало backend/app/schemas/dataset.py.

export interface Dataset {
  id: string;
  original_filename: string;
  file_size_bytes: number;
  format: "csv" | "xlsx";
  n_rows: number | null;
  n_cols: number | null;
  uploaded_at: string;
}

export interface DatasetPreview {
  columns: string[];
  rows: Array<Array<string | number | boolean | null>>;
  dtypes: Record<string, string>;
}

export interface DatasetWithPreview extends Dataset {
  preview: DatasetPreview;
}

// GET /api/datasets/{id}/usage (Спринт 6, Phase 4.2): счётчики связанных
// артефактов для confirm-диалога удаления. reports_count считает только
// success-PDF (failed/pending исключены).
export interface DatasetUsage {
  analyses_count: number;
  reports_count: number;
}
