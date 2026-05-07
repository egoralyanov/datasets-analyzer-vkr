// API-клиент админ-панели. Эндпоинты: GET /admin/stats, GET /admin/users,
// GET /admin/users/{id}, DELETE /admin/users/{id}. Все требуют
// user.role === "admin"; иначе бэк возвращает 403.
import { apiClient } from "./client";
import type {
  AdminStats,
  AdminUserDetail,
  AdminUserListResponse,
} from "../types/admin";

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

  async getUser(id: string): Promise<AdminUserDetail> {
    const res = await apiClient.get<AdminUserDetail>(`/admin/users/${id}`);
    return res.data;
  },

  // 204 — успех. 409 detail передаётся в DeleteUserConfirmModal:
  //   "Cannot delete your own admin account" — самоудаление,
  //   "Cannot delete the last admin account" — последний admin.
  async removeUser(id: string): Promise<void> {
    await apiClient.delete(`/admin/users/${id}`);
  },
};
