// Типы зеркальны Pydantic-схемам бэка (см. backend/app/schemas/user.py, auth.py).

export interface User {
  id: string;
  email: string;
  username: string;
  role: "user" | "admin";
  created_at: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface LoginRequest {
  username_or_email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface UpdateProfileRequest {
  email?: string;
  username?: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

// Личные счётчики юзера для дашборда (Sprint 6, Phase 7).
// Зеркало backend/app/schemas/user.py:UserStats — ответ GET /api/me/stats.
export interface UserStats {
  datasets_count: number;
  analyses_count: number;
  successful_analyses_count: number;
  reports_count: number;
}
