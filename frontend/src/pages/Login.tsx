// Страница входа.
//
// Стиль (Sprint 5, Phase 3.9): scientific/archive — §-заголовок, hairline-
// контейнер без скругления и теней, инпуты с тонкой ink-обводкой при focus
// (без bright-glow), archive-кнопка primary.
import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { authApi } from "../api/auth";
import { useAuthStore } from "../store/authStore";

interface LocationState {
  flash?: string;
}

export function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const login = useAuthStore((s) => s.login);

  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Зелёный flash после регистрации (передаётся через navigate state).
  const flash = (location.state as LocationState | null)?.flash ?? null;

  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await authApi.login({
        username_or_email: usernameOrEmail.trim(),
        password,
      });
      login(res.user, res.access_token);
      navigate("/");
    } catch (err: unknown) {
      const detail = extractDetail(err);
      setError(detail ?? "Не удалось войти. Проверьте логин и пароль.");
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-[480px] px-8 py-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        АУТЕНТИФИКАЦИЯ
      </p>
      <h1 className="mt-2 flex items-baseline gap-3 font-serif text-[2rem] font-bold leading-tight tracking-tight text-paper-900">
        <span className="font-sans text-base font-medium text-paper-400">
          §
        </span>
        Вход
      </h1>
      <p className="mt-3 font-serif text-sm leading-relaxed text-paper-600">
        Введите username или email и пароль.
      </p>

      <div className="mt-6 border border-paper-300 bg-paper-50 p-6">
        {flash && (
          <div className="mb-4 border-l-[3px] border-success-500 bg-paper-50 px-4 py-2 font-serif text-sm leading-relaxed text-paper-700">
            <span className="font-sans text-xs font-medium uppercase tracking-wider text-success-700">
              ✓ ОК ·
            </span>{" "}
            {flash}
          </div>
        )}
        {error && (
          <div className="mb-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
            {error}
          </div>
        )}

        <form className="space-y-5" onSubmit={onSubmit}>
          <Field label="Username или email" htmlFor="login-id">
            <input
              id="login-id"
              ref={inputRef}
              type="text"
              autoComplete="username"
              required
              value={usernameOrEmail}
              onChange={(e) => setUsernameOrEmail(e.target.value)}
              className="block w-full border border-paper-400 bg-paper-50 px-3 py-2 font-mono text-sm text-paper-800 placeholder:text-paper-400 focus:border-ink-700 focus:outline-none focus:ring-0"
            />
          </Field>

          <Field label="Пароль" htmlFor="login-password">
            <div className="relative">
              <input
                id="login-password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                required
                minLength={1}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full border border-paper-400 bg-paper-50 px-3 py-2 pr-10 font-mono text-sm text-paper-800 placeholder:text-paper-400 focus:border-ink-700 focus:outline-none focus:ring-0"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-paper-500 hover:text-ink-700"
                aria-label={showPassword ? "Скрыть пароль" : "Показать пароль"}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </Field>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex w-full items-center justify-center gap-2 border border-ink-700 bg-ink-700 px-4 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            ВОЙТИ
          </button>
        </form>
      </div>

      <p className="mt-5 font-serif text-sm text-paper-600">
        Нет аккаунта?{" "}
        <Link
          to="/register"
          className="border-b border-ink-700 pb-0.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700"
        >
          ЗАРЕГИСТРИРОВАТЬСЯ
        </Link>
      </p>
    </div>
  );
}

function Field({
  label,
  htmlFor,
  children,
}: {
  label: string;
  htmlFor: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        htmlFor={htmlFor}
        className="block font-sans text-xs font-medium uppercase tracking-wider text-paper-500"
      >
        {label}
      </label>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

// Возвращает осмысленное сообщение об ошибке из axios-исключения.
function extractDetail(err: unknown): string | null {
  if (
    typeof err === "object" &&
    err !== null &&
    "response" in err &&
    typeof (err as { response?: { data?: { detail?: unknown } } }).response?.data
      ?.detail === "string"
  ) {
    return (err as { response: { data: { detail: string } } }).response.data
      .detail;
  }
  return null;
}
