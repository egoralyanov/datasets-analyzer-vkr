// Шапка сайта с логотипом и навигацией. Показывается на всех страницах.
//
// Стиль (Sprint 5, Phase 2): scientific/archive — двойная рулька снизу,
// uppercase tracking-навигация, активная ссылка через подчёркивание, никаких
// иконок и pill-кнопок. См. frontend/DESIGN_TOKENS.md, раздел 8.1.
import { Link, NavLink, useNavigate } from "react-router-dom";
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
    <header
      className="border-b-[3px] border-double border-paper-400 bg-paper-50"
    >
      <div className="mx-auto flex max-w-[1100px] items-end justify-between gap-8 px-8 py-5 lg:px-16">
        <Link to="/" className="block leading-tight">
          <span className="block font-serif text-[1.75rem] font-bold tracking-tight text-paper-900">
            АНАЛИЗАТОР
          </span>
          <span className="mt-1 block font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
            Интеллектуальная система анализа наборов данных
          </span>
        </Link>

        <nav className="flex items-end gap-5 pb-1 text-xs font-sans font-medium uppercase tracking-wider">
          {user ? (
            <>
              <NavItem to="/upload" label="Загрузить" />
              <NavItem to="/history" label="История" />
              {user.role === "admin" && <NavItem to="/admin" label="Админ" />}
              <NavItem to="/profile" label={`@${user.username}`} mono />
              <button
                type="button"
                onClick={onLogout}
                className="border-b-2 border-transparent pb-0.5 text-paper-600 transition-colors hover:border-ink-700 hover:text-ink-700"
              >
                Выйти
              </button>
            </>
          ) : (
            <>
              <NavItem to="/login" label="Войти" />
              <NavItem to="/register" label="Регистрация" />
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

function NavItem({
  to,
  label,
  mono = false,
}: {
  to: string;
  label: string;
  mono?: boolean;
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        [
          mono ? "font-mono normal-case tracking-normal" : "",
          "border-b-2 pb-0.5 transition-colors",
          isActive
            ? "border-ink-700 text-ink-900"
            : "border-transparent text-paper-600 hover:text-ink-700",
        ]
          .filter(Boolean)
          .join(" ")
      }
    >
      {label}
    </NavLink>
  );
}
