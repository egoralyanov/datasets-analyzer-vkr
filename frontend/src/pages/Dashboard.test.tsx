// Smoke-тесты Dashboard (Sprint 6, Phase 7).
//
// Покрытие:
// - hero рендерит "С возвращением, {username}".
// - mini-stats скрыты при всех counts === 0 и отрисовываются при > 0.
// - empty state «Анализов пока нет» при пустом списке анализов.
// - empty state «Датасетов пока нет» при пустом списке датасетов.
// - финальный CTA с динамическим заголовком (есть датасеты vs нет).
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { Dashboard } from "./Dashboard";
import { useAuthStore } from "../store/authStore";

vi.mock("../api/me", () => ({
  meApi: {
    getStats: vi.fn(),
  },
}));

vi.mock("../api/analyses", () => ({
  analysesApi: {
    list: vi.fn(),
  },
}));

vi.mock("../api/datasets", () => ({
  datasetsApi: {
    list: vi.fn(),
  },
}));

import { meApi } from "../api/me";
import { analysesApi } from "../api/analyses";
import { datasetsApi } from "../api/datasets";

function renderDashboard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({
      user: {
        id: "u-1",
        email: "alice@example.com",
        username: "alice",
        role: "user",
        created_at: "2026-01-01T00:00:00Z",
      },
      token: "t-1",
    });
  });

  it("hero рендерит «С возвращением, {username}»", async () => {
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 0,
      analyses_count: 0,
      successful_analyses_count: 0,
      reports_count: 0,
    });
    vi.mocked(analysesApi.list).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 5,
      pages: 0,
    });
    vi.mocked(datasetsApi.list).mockResolvedValue([]);
    renderDashboard();
    expect(
      screen.getByRole("heading", { level: 1, name: /с возвращением, alice/i }),
    ).toBeInTheDocument();
  });

  it("mini-stats скрыты, когда все четыре counter'а = 0", async () => {
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 0,
      analyses_count: 0,
      successful_analyses_count: 0,
      reports_count: 0,
    });
    vi.mocked(analysesApi.list).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 5,
      pages: 0,
    });
    vi.mocked(datasetsApi.list).mockResolvedValue([]);
    renderDashboard();
    // Стат-загружается — ждём, пока запрос отрешится.
    await waitFor(() =>
      expect(meApi.getStats).toHaveBeenCalled(),
    );
    expect(screen.queryByText(/ваша статистика/i)).not.toBeInTheDocument();
  });

  it("mini-stats отрисовываются, когда есть хотя бы один counter > 0", async () => {
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 3,
      analyses_count: 12,
      successful_analyses_count: 9,
      reports_count: 5,
    });
    vi.mocked(analysesApi.list).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 5,
      pages: 0,
    });
    vi.mocked(datasetsApi.list).mockResolvedValue([]);
    renderDashboard();
    await waitFor(() =>
      expect(screen.getByText(/ваша статистика/i)).toBeInTheDocument(),
    );
    // Подсказка «из {analyses_count}» под success-карточкой.
    expect(screen.getByText(/из 12/i)).toBeInTheDocument();
  });

  it("показывает empty state на пустом списке анализов и датасетов", async () => {
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 0,
      analyses_count: 0,
      successful_analyses_count: 0,
      reports_count: 0,
    });
    vi.mocked(analysesApi.list).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 5,
      pages: 0,
    });
    vi.mocked(datasetsApi.list).mockResolvedValue([]);
    renderDashboard();
    await waitFor(() =>
      expect(screen.getByText(/анализов пока нет/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/датасетов пока нет/i)).toBeInTheDocument();
  });

  it("финальный CTA меняет заголовок в зависимости от наличия датасетов", async () => {
    // Сценарий 1: есть датасеты.
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 2,
      analyses_count: 5,
      successful_analyses_count: 4,
      reports_count: 1,
    });
    vi.mocked(analysesApi.list).mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 5,
      pages: 0,
    });
    vi.mocked(datasetsApi.list).mockResolvedValue([]);
    const { unmount } = renderDashboard();
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /анализ ещё одного датасета/i }),
      ).toBeInTheDocument(),
    );
    unmount();

    // Сценарий 2: датасетов нет — заголовок другой.
    vi.mocked(meApi.getStats).mockResolvedValue({
      datasets_count: 0,
      analyses_count: 0,
      successful_analyses_count: 0,
      reports_count: 0,
    });
    renderDashboard();
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: /готовы начать/i }),
      ).toBeInTheDocument(),
    );
  });
});
