// Защита админ-маршрутов: при role !== "admin" редиректит на /.
// Используется как nested wrapper под RequireAuth: гарантия user !== null
// уже даётся внешним guard'ом, здесь проверяем только роль.
import { Navigate, Outlet } from "react-router-dom";
import { useAuthStore } from "../../store/authStore";

export function RequireAdmin() {
  const user = useAuthStore((s) => s.user);

  if (user?.role !== "admin") {
    return <Navigate to="/" replace />;
  }
  return <Outlet />;
}
