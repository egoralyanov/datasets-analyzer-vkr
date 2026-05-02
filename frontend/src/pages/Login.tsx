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
    <div className="max-w-md mx-auto px-6 py-12">
      <div className="rounded-lg border border-slate-200 bg-white shadow-sm p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Вход</h1>
        <p className="mt-1 text-sm text-slate-600">
          Введите username или email и пароль.
        </p>

        {flash && (
          <div className="mt-4 rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-800">
            {flash}
          </div>
        )}
        {error && (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            {error}
          </div>
        )}

        <form className="mt-5 space-y-4" onSubmit={onSubmit}>
          <Field label="Username или email" htmlFor="login-id">
            <input
              id="login-id"
              ref={inputRef}
              type="text"
              autoComplete="username"
              required
              value={usernameOrEmail}
              onChange={(e) => setUsernameOrEmail(e.target.value)}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
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
                className="w-full rounded-md border border-slate-300 px-3 py-2 pr-10 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-slate-500 hover:text-slate-700"
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
            className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Войти
          </button>
        </form>

        <p className="mt-5 text-sm text-slate-600">
          Нет аккаунта?{" "}
          <Link to="/register" className="text-blue-600 hover:text-blue-700">
            Зарегистрироваться
          </Link>
        </p>
      </div>
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
      <label htmlFor={htmlFor} className="block text-sm font-medium text-slate-900">
        {label}
      </label>
      <div className="mt-1">{children}</div>
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
    return (err as { response: { data: { detail: string } } }).response.data.detail;
  }
  return null;
}
