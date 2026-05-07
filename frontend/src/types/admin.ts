// TypeScript-зеркала Pydantic-схем backend/app/schemas/admin.py.

export type AdminStats = {
  total_users: number;
  total_datasets: number;
  total_analyses: number;
  total_reports: number;
  // [0, 1] либо null если total соответствующей сущности = 0.
  analyses_success_rate: number | null;
  reports_success_rate: number | null;
};

export type AdminUserListItem = {
  id: string;
  email: string;
  username: string;
  role: "user" | "admin";
  created_at: string;
  datasets_count: number;
  analyses_count: number;
};

export type AdminUserListResponse = {
  items: AdminUserListItem[];
  total: number;
  page: number;
  size: number;
  pages: number;
};

// Детальная карточка пользователя (см. backend/app/schemas/admin.py,
// AdminUserDetail). reports_count считает только success-отчёты.
export type AdminUserDetail = {
  id: string;
  email: string;
  username: string;
  role: "user" | "admin";
  created_at: string;
  datasets_count: number;
  analyses_count: number;
  reports_count: number;
};
