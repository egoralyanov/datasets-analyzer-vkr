// Методы API для запуска и получения анализа.
// См. backend/app/api/analyses.py.
import { apiClient } from "./client";
import type {
  Analysis,
  AnalysisResult,
  StartAnalysisRequest,
} from "../types/analysis";

export const analysesApi = {
  async start(
    datasetId: string,
    params: StartAnalysisRequest = {},
  ): Promise<Analysis> {
    const res = await apiClient.post<Analysis>(
      `/datasets/${datasetId}/analyze`,
      params,
    );
    return res.data;
  },
  async get(id: string): Promise<Analysis> {
    const res = await apiClient.get<Analysis>(`/analyses/${id}`);
    return res.data;
  },
  async getResult(id: string): Promise<AnalysisResult> {
    const res = await apiClient.get<AnalysisResult>(`/analyses/${id}/result`);
    return res.data;
  },
  async list(): Promise<Analysis[]> {
    const res = await apiClient.get<Analysis[]>("/analyses");
    return res.data;
  },
};
