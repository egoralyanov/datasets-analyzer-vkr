// Страница профиля. Три режима: просмотр, редактирование, смена пароля.
//
// Стиль (Sprint 5, Phase 3.8): scientific/archive — §-заголовок, специмен-
// definition list для просмотра, archive-кнопки и архивные инпуты.
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { authApi } from "../api/auth";
import { useAuthStore } from "../store/authStore";

type Mode = "view" | "edit-profile" | "change-password";

export function Profile() {
  const user = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  const [mode, setMode] = useState<Mode>("view");
  const [flash, setFlash] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!user) return null; // защищено RequireAuth, но TS требует.

  return (
    <div className="mx-auto max-w-[760px] px-8 py-12 lg:px-16">
      <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        АККАУНТ
      </p>
      <h1 className="mt-2 flex items-baseline gap-3 font-serif text-[2.25rem] font-bold leading-tight tracking-tight text-paper-900">
        <span className="font-sans text-base font-medium text-paper-400">
          §
        </span>
        Профиль
      </h1>

      {flash && (
        <div className="mt-6 border-l-[3px] border-success-500 bg-paper-50 px-4 py-2 font-serif text-sm leading-relaxed text-paper-700">
          <span className="font-sans text-xs font-medium uppercase tracking-wider text-success-700">
            ✓ ОК ·
          </span>{" "}
          {flash}
        </div>
      )}
      {error && (
        <div className="mt-6 border-l-[3px] border-critical-500 bg-critical-50/70 px-4 py-2 font-sans text-sm text-critical-700">
          {error}
        </div>
      )}

      <div className="mt-8">
        {mode === "view" && (
          <ViewMode
            onEdit={() => {
              setMode("edit-profile");
              setError(null);
              setFlash(null);
            }}
            onChangePassword={() => {
              setMode("change-password");
              setError(null);
              setFlash(null);
            }}
          />
        )}
        {mode === "edit-profile" && (
          <EditProfileForm
            onCancel={() => setMode("view")}
            onSaved={(u) => {
              setUser(u);
              setMode("view");
              setFlash("Профиль обновлён.");
              setError(null);
            }}
            onError={(msg) => {
              setError(msg);
              setFlash(null);
            }}
          />
        )}
        {mode === "change-password" && (
          <ChangePasswordForm
            onCancel={() => setMode("view")}
            onSaved={() => {
              setMode("view");
              setFlash(
                "Пароль обновлён. Старые сессии остаются активными до истечения токена.",
              );
              setError(null);
            }}
            onError={(msg) => {
              setError(msg);
              setFlash(null);
            }}
          />
        )}
      </div>
    </div>
  );
}

function ViewMode({
  onEdit,
  onChangePassword,
}: {
  onEdit: () => void;
  onChangePassword: () => void;
}) {
  const user = useAuthStore((s) => s.user)!;
  return (
    <>
      <div className="border border-paper-300 bg-paper-50">
        <dl className="divide-y divide-paper-200">
          <Row label="Email" value={user.email} mono />
          <Row label="Username" value={user.username} mono />
          <Row label="Роль" value={user.role.toUpperCase()} mono />
          <Row
            label="Зарегистрирован"
            value={new Date(user.created_at).toLocaleString("ru-RU")}
            mono
          />
        </dl>
      </div>
      <div className="mt-6 flex flex-wrap gap-2">
        <ArchiveButton onClick={onEdit} variant="secondary">
          ИЗМЕНИТЬ ПРОФИЛЬ
        </ArchiveButton>
        <ArchiveButton onClick={onChangePassword} variant="secondary">
          СМЕНИТЬ ПАРОЛЬ
        </ArchiveButton>
      </div>
    </>
  );
}

function Row({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="grid grid-cols-[12rem_1fr] gap-6 px-5 py-3">
      <dt className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
        {label}
      </dt>
      <dd
        className={`break-all text-sm text-paper-800 ${
          mono ? "font-mono" : "font-sans"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

function EditProfileForm({
  onCancel,
  onSaved,
  onError,
}: {
  onCancel: () => void;
  onSaved: (u: import("../types/user").User) => void;
  onError: (msg: string) => void;
}) {
  const user = useAuthStore((s) => s.user)!;
  const [email, setEmail] = useState(user.email);
  const [username, setUsername] = useState(user.username);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const updated = await authApi.updateMe({
        email: email !== user.email ? email : undefined,
        username: username !== user.username ? username : undefined,
      });
      onSaved(updated);
    } catch (err: unknown) {
      onError(extractDetail(err) ?? "Не удалось сохранить профиль.");
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-5">
      <Field label="Email" htmlFor="edit-email">
        <ArchiveInput
          id="edit-email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </Field>
      <Field label="Username" htmlFor="edit-username">
        <ArchiveInput
          id="edit-username"
          type="text"
          autoComplete="username"
          required
          minLength={3}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </Field>
      <div className="flex gap-2">
        <ArchiveButton type="submit" disabled={loading}>
          {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          СОХРАНИТЬ
        </ArchiveButton>
        <ArchiveButton
          type="button"
          onClick={onCancel}
          disabled={loading}
          variant="secondary"
        >
          ОТМЕНА
        </ArchiveButton>
      </div>
    </form>
  );
}

function ChangePasswordForm({
  onCancel,
  onSaved,
  onError,
}: {
  onCancel: () => void;
  onSaved: () => void;
  onError: (msg: string) => void;
}) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword.length < 8) {
      onError("Новый пароль должен содержать минимум 8 символов.");
      return;
    }
    if (newPassword !== confirm) {
      onError("Пароли не совпадают.");
      return;
    }
    setLoading(true);
    try {
      await authApi.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      onSaved();
    } catch (err: unknown) {
      onError(extractDetail(err) ?? "Не удалось сменить пароль.");
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-5">
      <Field label="Текущий пароль" htmlFor="cp-current">
        <ArchiveInput
          id="cp-current"
          type="password"
          autoComplete="current-password"
          required
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
        />
      </Field>
      <Field label="Новый пароль" htmlFor="cp-new">
        <ArchiveInput
          id="cp-new"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />
      </Field>
      <Field label="Повторите новый пароль" htmlFor="cp-confirm">
        <ArchiveInput
          id="cp-confirm"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
        />
      </Field>
      <div className="flex gap-2">
        <ArchiveButton type="submit" disabled={loading}>
          {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          СМЕНИТЬ ПАРОЛЬ
        </ArchiveButton>
        <ArchiveButton
          type="button"
          onClick={onCancel}
          disabled={loading}
          variant="secondary"
        >
          ОТМЕНА
        </ArchiveButton>
      </div>
    </form>
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

function ArchiveInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className="block w-full border border-paper-400 bg-paper-50 px-3 py-2 font-mono text-sm text-paper-800 placeholder:text-paper-400 focus:border-ink-700 focus:outline-none focus:ring-0"
    />
  );
}

function ArchiveButton({
  children,
  type = "button",
  onClick,
  disabled = false,
  variant = "primary",
}: {
  children: React.ReactNode;
  type?: "button" | "submit";
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}) {
  const cls =
    variant === "primary"
      ? "border border-ink-700 bg-ink-700 text-paper-50 hover:bg-ink-800"
      : "border border-ink-700 bg-paper-50 text-ink-700 hover:bg-ink-700 hover:text-paper-50";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-2 px-4 py-2 font-sans text-xs font-medium uppercase tracking-wider transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${cls}`}
    >
      {children}
    </button>
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
    return (err as { response: { data: { detail: string } } }).response.data
      .detail;
  }
  return null;
}
