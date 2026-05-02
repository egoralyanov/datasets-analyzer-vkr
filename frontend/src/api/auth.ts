// Методы API аутентификации. См. .knowledge/architecture/api-contract.md, раздел 1.
import { apiClient } from "./client";
import type {
  ChangePasswordRequest,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  UpdateProfileRequest,
  User,
} from "../types/user";

export const authApi = {
  async register(data: RegisterRequest): Promise<User> {
    const res = await apiClient.post<User>("/auth/register", data);
    return res.data;
  },
  async login(data: LoginRequest): Promise<LoginResponse> {
    const res = await apiClient.post<LoginResponse>("/auth/login", data);
    return res.data;
  },
  async logout(): Promise<void> {
    await apiClient.post("/auth/logout");
  },
  async getMe(): Promise<User> {
    const res = await apiClient.get<User>("/auth/me");
    return res.data;
  },
  async updateMe(data: UpdateProfileRequest): Promise<User> {
    const res = await apiClient.put<User>("/auth/me", data);
    return res.data;
  },
  async changePassword(data: ChangePasswordRequest): Promise<void> {
    await apiClient.put("/auth/password", data);
  },
};
