// Методы API для запуска и получения анализа.
// См. backend/app/api/analyses.py.
import { apiClient } from "./client";
import type {
  Analysis,
  AnalysisListResponse,
  AnalysisResult,
  AnalysisStatus,
  BaselineResponse,
  BaselineStartResponse,
  SimilarDataset,
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
  // Пагинированный список для страницы /history. Контракт: {items, total,
  // page, size, pages}. Дефолты бэка: page=1, size=20, status=null.
  async list(
    params: {
      page?: number;
      size?: number;
      status?: AnalysisStatus | null;
    } = {},
  ): Promise<AnalysisListResponse> {
    const query: Record<string, string | number> = {};
    if (params.page !== undefined) query.page = params.page;
    if (params.size !== undefined) query.size = params.size;
    if (params.status) query.status = params.status;
    const res = await apiClient.get<AnalysisListResponse>("/analyses", {
      params: query,
    });
    return res.data;
  },
  // POST: 202 для свежего запуска, 200 если baseline уже done (идемпотентность).
  async startBaseline(id: string): Promise<BaselineStartResponse> {
    const res = await apiClient.post<BaselineStartResponse>(
      `/analyses/${id}/baseline`,
    );
    return res.data;
  },
  // GET: 404 если baseline_status='not_started' — фронт это перехватывает
  // и трактует как «обучения не было».
  async getBaseline(id: string): Promise<BaselineResponse> {
    const res = await apiClient.get<BaselineResponse>(
      `/analyses/${id}/baseline`,
    );
    return res.data;
  },
  async getSimilar(
    id: string,
    topK: number = 5,
  ): Promise<SimilarDataset[]> {
    const res = await apiClient.get<SimilarDataset[]>(
      `/analyses/${id}/similar`,
      { params: { top_k: topK } },
    );
    return res.data;
  },
};
