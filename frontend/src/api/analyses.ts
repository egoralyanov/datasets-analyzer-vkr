// Методы API для запуска и получения анализа.
// См. backend/app/api/analyses.py.
import { apiClient } from "./client";
import type {
  Analysis,
  AnalysisResult,
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
  async list(): Promise<Analysis[]> {
    const res = await apiClient.get<Analysis[]>("/analyses");
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
