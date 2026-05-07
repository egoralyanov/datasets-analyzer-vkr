// Личные эндпоинты текущего пользователя (Sprint 6, Phase 7).
// См. backend/app/api/me.py.
import { apiClient } from "./client";
import type { UserStats } from "../types/user";

export const meApi = {
  // 200 + UserStats. 401 → axios-interceptor в client.ts чистит auth-state
  // и редиректит на /login.
  async getStats(): Promise<UserStats> {
    const res = await apiClient.get<UserStats>("/me/stats");
    return res.data;
  },
};
