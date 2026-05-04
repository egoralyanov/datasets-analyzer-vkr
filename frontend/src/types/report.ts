// TypeScript-зеркала Pydantic-схем из backend/app/schemas/report.py.

export type ReportStatus = "pending" | "running" | "success" | "failed";

export type Report = {
  id: string;
  analysis_id: string;
  status: ReportStatus;
  file_size_bytes: number | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type ReportCreateResponse = {
  id: string;
  status: ReportStatus;
};

// Тело ответа 409 при конфликте создания отчёта. Поле reason —
// машинно-читаемое: фронт по нему различает два кейса (см. ReportDownloadCard).
export type ReportConflictResponse = {
  detail: string;
  reason: "analysis_not_done" | "report_in_progress";
  report_id: string | null;
  status: ReportStatus | null;
};
