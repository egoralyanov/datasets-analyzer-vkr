// Render-тесты DeleteDatasetModal (Sprint 6, Phase 5.1).
//
// Покрытие:
// - Заголовок содержит filename.
// - Тело при usage > 0 — про N анализов и M отчётов с правильным склонением.
// - Тело при usage 0/0 — про «Анализов и отчётов нет».
// - Loading-плейсхолдер показывается до прихода usage.
//
// API-вызов datasetsApi.getUsage мокаем через vi.mock — тесты не должны
// делать сетевых запросов. React Query в тестах — обычный QueryClient
// без ретраев и кеша между тестами.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { DeleteDatasetModal } from "./DeleteDatasetModal";

vi.mock("../../api/datasets", () => ({
  datasetsApi: {
    getUsage: vi.fn(),
    remove: vi.fn(),
  },
}));

import { datasetsApi } from "../../api/datasets";

function renderModal(props: Partial<Parameters<typeof DeleteDatasetModal>[0]> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <DeleteDatasetModal
        open
        datasetId="ds-1"
        filename="iris.csv"
        onClose={() => {}}
        {...props}
      />
    </QueryClientProvider>,
  );
}

describe("DeleteDatasetModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("рендерит заголовок с filename", () => {
    vi.mocked(datasetsApi.getUsage).mockResolvedValue({
      analyses_count: 0,
      reports_count: 0,
    });
    renderModal();
    expect(
      screen.getByRole("heading", { name: "Удалить датасет «iris.csv»?" }),
    ).toBeInTheDocument();
  });

  it("показывает плейсхолдер пока usage грузится", () => {
    vi.mocked(datasetsApi.getUsage).mockReturnValue(new Promise(() => {}));
    renderModal();
    expect(
      screen.getByText(/считаем связанные артефакты/i),
    ).toBeInTheDocument();
  });

  it("при usage > 0 показывает текст с правильным склонением", async () => {
    vi.mocked(datasetsApi.getUsage).mockResolvedValue({
      analyses_count: 2,
      reports_count: 5,
    });
    renderModal();
    await waitFor(() =>
      expect(
        screen.getByText(/также будет удалено/i),
      ).toBeInTheDocument(),
    );
    // 2 → «анализа», 5 → «отчётов».
    const body = screen.getByText(/также будет удалено/i);
    expect(body.textContent).toContain("2 анализа");
    expect(body.textContent).toContain("5 отчётов");
    expect(body.textContent).toContain("Действие необратимо");
  });

  it("при usage = 1/1 показывает единственное число", async () => {
    vi.mocked(datasetsApi.getUsage).mockResolvedValue({
      analyses_count: 1,
      reports_count: 1,
    });
    renderModal();
    await waitFor(() => {
      const body = screen.getByText(/также будет удалено/i);
      expect(body.textContent).toContain("1 анализ");
      expect(body.textContent).toContain("1 отчёт");
    });
  });

  it("при usage 0/0 показывает «нет анализов и отчётов»", async () => {
    vi.mocked(datasetsApi.getUsage).mockResolvedValue({
      analyses_count: 0,
      reports_count: 0,
    });
    renderModal();
    await waitFor(() =>
      expect(
        screen.getByText(/анализов и отчётов нет/i),
      ).toBeInTheDocument(),
    );
  });

  it("кнопка УДАЛИТЬ доступна сразу при открытии (даже до загрузки usage)", () => {
    vi.mocked(datasetsApi.getUsage).mockReturnValue(new Promise(() => {}));
    renderModal();
    const deleteBtn = screen.getByRole("button", { name: /удалить/i });
    expect(deleteBtn).not.toBeDisabled();
  });
});
