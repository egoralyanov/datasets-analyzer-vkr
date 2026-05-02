import { useState } from "react";
import { Loader2, Mail, User as UserIcon, Lock } from "lucide-react";
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
    <div className="max-w-2xl mx-auto px-6 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Профиль</h1>

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

      <div className="mt-6 rounded-lg border border-slate-200 bg-white shadow-sm p-6">
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
              setFlash("Пароль обновлён. Старые сессии остаются активными до истечения токена.");
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
      <dl className="space-y-4">
        <Row icon={<Mail className="h-4 w-4 text-slate-500" />} label="Email" value={user.email} />
        <Row icon={<UserIcon className="h-4 w-4 text-slate-500" />} label="Username" value={user.username} />
        <Row label="Роль" value={user.role} />
        <Row
          label="Зарегистрирован"
          value={new Date(user.created_at).toLocaleString("ru-RU")}
        />
      </dl>
      <div className="mt-6 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onEdit}
          className="inline-flex items-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          Изменить профиль
        </button>
        <button
          type="button"
          onClick={onChangePassword}
          className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          <Lock className="h-4 w-4" />
          Сменить пароль
        </button>
      </div>
    </>
  );
}

function Row({
  icon,
  label,
  value,
}: {
  icon?: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <dt className="w-40 text-sm text-slate-500 flex items-center gap-2">
        {icon}
        {label}
      </dt>
      <dd className="text-sm text-slate-900 break-all">{value}</dd>
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
    <form onSubmit={onSubmit} className="space-y-4">
      <Field label="Email" htmlFor="edit-email">
        <input
          id="edit-email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
        />
      </Field>
      <Field label="Username" htmlFor="edit-username">
        <input
          id="edit-username"
          type="text"
          autoComplete="username"
          required
          minLength={3}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
        />
      </Field>
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Сохранить
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={loading}
          className="inline-flex items-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          Отмена
        </button>
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
    <form onSubmit={onSubmit} className="space-y-4">
      <Field label="Текущий пароль" htmlFor="cp-current">
        <input
          id="cp-current"
          type="password"
          autoComplete="current-password"
          required
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
        />
      </Field>
      <Field label="Новый пароль" htmlFor="cp-new">
        <input
          id="cp-new"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
        />
      </Field>
      <Field label="Повторите новый пароль" htmlFor="cp-confirm">
        <input
          id="cp-confirm"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-600"
        />
      </Field>
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Сменить пароль
        </button>
        <button
          type="button"
          onClick={onCancel}
          disabled={loading}
          className="inline-flex items-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          Отмена
        </button>
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
