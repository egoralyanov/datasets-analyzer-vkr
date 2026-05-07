// Бейдж роли пользователя в админ-листинге и в карточке деталей
// (Sprint 6, Phase 5.4). Стилистика повторяет inline-вариант, который
// жил в Admin.tsx: тонкая обводка + uppercase tracking-wider, без заливки.
//   - admin → border-info-500 + text-info-700
//   - user  → border-paper-400 + text-paper-600
type Role = "admin" | "user";

type Props = {
  role: Role;
};

export function RolePill({ role }: Props) {
  const isAdmin = role === "admin";
  return (
    <span
      className={`inline-block border px-2 py-0.5 font-sans text-[0.6875rem] font-medium uppercase tracking-wider ${
        isAdmin
          ? "border-info-500 text-info-700"
          : "border-paper-400 text-paper-600"
      }`}
    >
      {isAdmin ? "ADMIN" : "USER"}
    </span>
  );
}
