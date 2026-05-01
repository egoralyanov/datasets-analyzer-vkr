import { apiClient } from "./client";
import type { HealthResponse } from "../types/health";

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/health");
  return data;
}
