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
