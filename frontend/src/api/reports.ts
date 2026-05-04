// API-клиент PDF-отчётов. Эндпоинты: POST /analyses/{id}/report,
// GET /reports/{id}, GET /reports/{id}/download.
import { apiClient } from "./client";
import type { Report, ReportCreateResponse } from "../types/report";

export const reportsApi = {
  create: async (analysisId: string): Promise<ReportCreateResponse> => {
    const { data } = await apiClient.post<ReportCreateResponse>(
      `/analyses/${analysisId}/report`,
    );
    return data;
  },

  get: async (reportId: string): Promise<Report> => {
    const { data } = await apiClient.get<Report>(`/reports/${reportId}`);
    return data;
  },

  // Скачивание: возвращает blob и распарсенное имя файла из
  // Content-Disposition. Имя нужно ставить в <a download={...}>, иначе
  // браузер либо откроет PDF во вкладке (без download-атрибута), либо
  // потеряет серверное имя (download="" игнорирует filename* из заголовка).
  download: async (
    reportId: string,
  ): Promise<{ blob: Blob; filename: string }> => {
    const response = await apiClient.get(`/reports/${reportId}/download`, {
      responseType: "blob",
    });
    const filename = parseContentDispositionFilename(
      response.headers["content-disposition"],
    );
    return { blob: response.data as Blob, filename };
  },
};

// Парсинг имени файла из Content-Disposition по RFC 5987.
// Сначала пробуем filename*=UTF-8''<urlencoded> (для современных браузеров,
// поддерживает кириллицу), потом ASCII-fallback filename="<name>", в крайнем
// случае — родовое "report.pdf". Экспортируется ради возможного юнит-теста.
export function parseContentDispositionFilename(
  header: string | undefined | null,
): string {
  if (!header) return "report.pdf";
  const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return "report.pdf";
    }
  }
  const asciiMatch = header.match(/filename="([^"]+)"/i);
  if (asciiMatch) return asciiMatch[1];
  return "report.pdf";
}
