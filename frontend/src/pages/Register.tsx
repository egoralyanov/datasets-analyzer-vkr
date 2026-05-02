import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { authApi } from "../api/auth";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_MIN = 3;
const PASSWORD_MIN = 8;

export function Register() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const validateLocal = (): string | null => {
    if (!EMAIL_RE.test(email)) return "Неверный формат email.";
    if (username.length < USERNAME_MIN)
      return `Username должен содержать минимум ${USERNAME_MIN} символа.`;
    if (password.length < PASSWORD_MIN)
      return `Пароль должен содержать минимум ${PASSWORD_MIN} символов.`;
    if (password !== confirm) return "Пароли не совпадают.";
    return null;
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const localError = validateLocal();
    if (localError) {
      setError(localError);
      return;
    }
    setLoading(true);
    try {
      await authApi.register({ email: email.trim(), username: username.trim(), password });
      navigate("/login", {
        state: { flash: "Регистрация успешна. Теперь войдите." },
      });
    } catch (err: unknown) {
      const detail = extractDetail(err);
      setError(detail ?? "Не удалось зарегистрироваться. Попробуйте ещё раз.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto px-6 py-12">
      <div className="rounded-lg border border-slate-200 bg-white shadow-sm p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Регистрация</h1>
        <p className="mt-1 text-sm text-slate-600">
          Создайте аккаунт, чтобы загружать датасеты и видеть историю анализов.
        </p>

        {error && (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            {error}
          </div>
        )}

        <form className="mt-5 space-y-4" onSubmit={onSubmit}>
          <Field label="Email" htmlFor="reg-email">
            <input
              id="reg-email"
              ref={inputRef}
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
            />
          </Field>

          <Field label="Username" htmlFor="reg-username">
            <input
              id="reg-username"
              type="text"
              autoComplete="username"
              required
              minLength={USERNAME_MIN}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
            />
          </Field>

          <Field label="Пароль" htmlFor="reg-password">
            <div className="relative">
              <input
                id="reg-password"
                type={showPassword ? "text" : "password"}
                autoComplete="new-password"
                required
                minLength={PASSWORD_MIN}
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
            <p className="mt-1 text-xs text-slate-500">Минимум {PASSWORD_MIN} символов.</p>
          </Field>

          <Field label="Повторите пароль" htmlFor="reg-confirm">
            <input
              id="reg-confirm"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              required
              minLength={PASSWORD_MIN}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
            />
          </Field>

          <button
            type="submit"
            disabled={loading}
            className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            Зарегистрироваться
          </button>
        </form>

        <p className="mt-5 text-sm text-slate-600">
          Уже есть аккаунт?{" "}
          <Link to="/login" className="text-blue-600 hover:text-blue-700">
            Войти
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
