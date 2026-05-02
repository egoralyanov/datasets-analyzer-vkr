// Методы API для работы с датасетами. См. .knowledge/architecture/api-contract.md, раздел 2.
import { apiClient } from "./client";
import type { Dataset, DatasetWithPreview } from "../types/dataset";

export const datasetsApi = {
  async upload(
    file: File,
    onProgress?: (percent: number) => void,
  ): Promise<DatasetWithPreview> {
    const form = new FormData();
    form.append("file", file);
    const res = await apiClient.post<DatasetWithPreview>(
      "/datasets/upload",
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        // Большие CSV/XLSX могут идти дольше дефолтного timeout 30s.
        timeout: 120_000,
        onUploadProgress: (evt) => {
          if (onProgress && evt.total) {
            onProgress(Math.round((evt.loaded / evt.total) * 100));
          }
        },
      },
    );
    return res.data;
  },
  async list(): Promise<Dataset[]> {
    const res = await apiClient.get<Dataset[]>("/datasets");
    return res.data;
  },
  async get(id: string): Promise<DatasetWithPreview> {
    const res = await apiClient.get<DatasetWithPreview>(`/datasets/${id}`);
    return res.data;
  },
  async remove(id: string): Promise<void> {
    await apiClient.delete(`/datasets/${id}`);
  },
};
