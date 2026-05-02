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
