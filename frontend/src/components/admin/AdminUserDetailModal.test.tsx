// Render-тесты AdminUserDetailModal (Sprint 6, Phase 5.4).
//
// Покрытие:
// - Загрузка → плейсхолдер «Загрузка пользователя…», title="Загрузка…".
// - Готовые данные → username в title, email/role/счётчики в body.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { AdminUserDetailModal } from "./AdminUserDetailModal";

vi.mock("../../api/admin", () => ({
  adminApi: {
    getUser: vi.fn(),
    removeUser: vi.fn(),
  },
}));

import { adminApi } from "../../api/admin";

function renderModal(
  props: Partial<Parameters<typeof AdminUserDetailModal>[0]> = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AdminUserDetailModal
        open
        onClose={() => {}}
        userId="u-1"
        {...props}
      />
    </QueryClientProvider>,
  );
}

describe("AdminUserDetailModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("показывает «Загрузка…» в title и спиннер в body, пока запрос идёт", () => {
    vi.mocked(adminApi.getUser).mockReturnValue(new Promise(() => {}));
    renderModal();
    expect(
      screen.getByRole("heading", { name: /загрузка/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/загрузка пользователя/i),
    ).toBeInTheDocument();
  });

  it("после успешной загрузки рендерит username/email/счётчики", async () => {
    vi.mocked(adminApi.getUser).mockResolvedValue({
      id: "u-1",
      email: "alice@example.com",
      username: "alice",
      role: "user",
      created_at: "2026-04-12T10:00:00Z",
      datasets_count: 3,
      analyses_count: 12,
      reports_count: 5,
    });
    renderModal();
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: "alice" })).toBeInTheDocument(),
    );
    expect(screen.getByText("alice@example.com")).toBeInTheDocument();
    // Счётчики выводятся числами с локалью; для маленьких значений
    // toLocaleString просто стрингифицирует — проверяем по точному совпадению.
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    // Кнопка действия видна.
    expect(
      screen.getByRole("button", { name: /удалить пользователя/i }),
    ).toBeInTheDocument();
  });
});
