// Шапка сайта с логотипом и навигацией. Показывается на всех страницах.
import { Link, useNavigate } from "react-router-dom";
import { LogOut, User as UserIcon } from "lucide-react";
import { useAuthStore } from "../../store/authStore";
import { authApi } from "../../api/auth";

export function Header() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const navigate = useNavigate();

  const onLogout = async () => {
    try {
      await authApi.logout();
    } catch {
      // 204 либо ошибка — для нас не критично, всё равно чистим локально.
    }
    logout();
    navigate("/");
  };

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="text-lg font-semibold text-slate-900">
          Анализатор
        </Link>
        <nav className="flex items-center gap-3">
          {user ? (
            <>
              <Link
                to="/upload"
                className="text-sm text-slate-700 hover:text-slate-900"
              >
                Мои датасеты
              </Link>
              <Link
                to="/history"
                className="text-sm text-slate-700 hover:text-slate-900"
              >
                История
              </Link>
              <Link
                to="/profile"
                className="inline-flex items-center gap-2 text-sm text-slate-700 hover:text-slate-900"
              >
                <UserIcon className="h-4 w-4" />
                {user.username}
              </Link>
              <button
                type="button"
                onClick={onLogout}
                className="inline-flex items-center gap-2 text-sm text-slate-700 hover:text-slate-900"
              >
                <LogOut className="h-4 w-4" />
                Выйти
              </button>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className="text-sm text-slate-700 hover:text-slate-900"
              >
                Войти
              </Link>
              <Link
                to="/register"
                className="inline-flex items-center rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
              >
                Регистрация
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
