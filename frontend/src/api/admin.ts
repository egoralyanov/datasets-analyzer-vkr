// API-клиент админ-панели. Эндпоинты: GET /admin/stats, GET /admin/users.
// Оба требуют user.role === "admin"; иначе бэк возвращает 403.
import { apiClient } from "./client";
import type { AdminStats, AdminUserListResponse } from "../types/admin";

export const adminApi = {
  async getStats(): Promise<AdminStats> {
    const res = await apiClient.get<AdminStats>("/admin/stats");
    return res.data;
  },

  async listUsers(
    params: { page?: number; size?: number } = {},
  ): Promise<AdminUserListResponse> {
    const query: Record<string, number> = {};
    if (params.page !== undefined) query.page = params.page;
    if (params.size !== undefined) query.size = params.size;
    const res = await apiClient.get<AdminUserListResponse>("/admin/users", {
      params: query,
    });
    return res.data;
  },
};
