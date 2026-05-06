// Страница регистрации.
//
// Стиль (Sprint 5, Phase 3.9): scientific/archive — §-заголовок, hairline-
// контейнер, инпуты с тонкой ink-обводкой при focus, archive-кнопка primary.
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
      await authApi.register({
        email: email.trim(),
        username: username.trim(),
        password,
      });
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
    <div className="mx-auto max-w-[480px] px-8 py-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        НОВЫЙ АККАУНТ
      </p>
      <h1 className="mt-2 flex items-baseline gap-3 font-serif text-[2rem] font-bold leading-tight tracking-tight text-paper-900">
        <span className="font-sans text-base font-medium text-paper-400">
          §
        </span>
        Регистрация
      </h1>
      <p className="mt-3 font-serif text-sm leading-relaxed text-paper-600">
        Создайте аккаунт, чтобы загружать датасеты и видеть историю анализов.
      </p>

      <div className="mt-6 border border-paper-300 bg-paper-50 p-6">
        {error && (
          <div className="mb-4 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
            {error}
          </div>
        )}

        <form className="space-y-5" onSubmit={onSubmit}>
          <Field label="Email" htmlFor="reg-email">
            <ArchiveInput
              id="reg-email"
              ref={inputRef}
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </Field>

          <Field label="Username" htmlFor="reg-username">
            <ArchiveInput
              id="reg-username"
              type="text"
              autoComplete="username"
              required
              minLength={USERNAME_MIN}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </Field>

          <Field label="Пароль" htmlFor="reg-password">
            <div className="relative">
              <ArchiveInput
                id="reg-password"
                type={showPassword ? "text" : "password"}
                autoComplete="new-password"
                required
                minLength={PASSWORD_MIN}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pr-10"
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
            <p className="mt-1 font-mono text-xs text-paper-500">
              минимум {PASSWORD_MIN} символов
            </p>
          </Field>

          <Field label="Повторите пароль" htmlFor="reg-confirm">
            <ArchiveInput
              id="reg-confirm"
              type={showPassword ? "text" : "password"}
              autoComplete="new-password"
              required
              minLength={PASSWORD_MIN}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          </Field>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex w-full items-center justify-center gap-2 border border-ink-700 bg-ink-700 px-4 py-2.5 font-sans text-xs font-medium uppercase tracking-wider text-paper-50 transition-colors hover:bg-ink-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            ЗАРЕГИСТРИРОВАТЬСЯ
          </button>
        </form>
      </div>

      <p className="mt-5 font-serif text-sm text-paper-600">
        Уже есть аккаунт?{" "}
        <Link
          to="/login"
          className="border-b border-ink-700 pb-0.5 font-sans text-xs font-medium uppercase tracking-wider text-ink-700"
        >
          ВОЙТИ
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

const ArchiveInput = ({
  className = "",
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement> & {
  ref?: React.Ref<HTMLInputElement>;
}) => (
  <input
    {...rest}
    className={`block w-full border border-paper-400 bg-paper-50 px-3 py-2 font-mono text-sm text-paper-800 placeholder:text-paper-400 focus:border-ink-700 focus:outline-none focus:ring-0 ${className}`}
  />
);

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
