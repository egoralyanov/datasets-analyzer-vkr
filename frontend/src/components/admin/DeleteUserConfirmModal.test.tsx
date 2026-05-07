// Render-тесты DeleteUserConfirmModal (Sprint 6, Phase 5.4).
//
// Покрытие:
// - Pluralize для counts > 0 — текст «N датасетов, M анализов, K PDF-отчётов».
// - Все counts === 0 → «У пользователя нет датасетов, анализов и отчётов».
// - 409 «your own admin account» → toast.warning «собственный аккаунт».
// - 409 «last admin account» → toast.warning «последнего администратора».
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AxiosError } from "axios";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { DeleteUserConfirmModal } from "./DeleteUserConfirmModal";

vi.mock("../../api/admin", () => ({
  adminApi: {
    removeUser: vi.fn(),
  },
}));

vi.mock("../../lib/toast", () => ({
  toast: {
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}));

import { adminApi } from "../../api/admin";
import { toast } from "../../lib/toast";

function renderModal(
  props: Partial<Parameters<typeof DeleteUserConfirmModal>[0]> = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const defaults = {
    open: true,
    onClose: vi.fn(),
    userId: "u-1",
    username: "alice",
    datasetsCount: 0,
    analysesCount: 0,
    reportsCount: 0,
  };
  const merged = { ...defaults, ...props };
  return {
    ...merged,
    ...render(
      <QueryClientProvider client={client}>
        <DeleteUserConfirmModal {...merged} />
      </QueryClientProvider>,
    ),
  };
}

// Конструируем AxiosError с {detail} в response.data — удобно для onError-веток.
function fakeAxios409(detail: string): AxiosError {
  const err = new AxiosError(detail, "ERR_BAD_REQUEST");
  err.response = {
    data: { detail },
    status: 409,
    statusText: "Conflict",
    headers: {},
    config: {} as never,
  };
  return err;
}

describe("DeleteUserConfirmModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("выводит pluralize для counts > 0", () => {
    renderModal({
      datasetsCount: 2,
      analysesCount: 5,
      reportsCount: 1,
    });
    const body = screen.getByText(/будут удалены/i);
    expect(body.textContent).toContain("2 датасета");
    expect(body.textContent).toContain("5 анализов");
    expect(body.textContent).toContain("1 PDF-отчёт");
    expect(body.textContent).toContain("Действие необратимо");
  });

  it("при всех counts=0 — текст «У пользователя нет…»", () => {
    renderModal({
      datasetsCount: 0,
      analysesCount: 0,
      reportsCount: 0,
    });
    expect(
      screen.getByText(/у пользователя нет датасетов, анализов и отчётов/i),
    ).toBeInTheDocument();
  });

  it("409 «your own admin account» → toast.warning о собственном аккаунте, onClose", async () => {
    vi.mocked(adminApi.removeUser).mockRejectedValue(
      fakeAxios409("Cannot delete your own admin account"),
    );
    const onClose = vi.fn();
    renderModal({ onClose });

    fireEvent.click(screen.getByRole("button", { name: /^удалить$/i }));

    await waitFor(() =>
      expect(toast.warning).toHaveBeenCalledWith(
        "Нельзя удалить собственный аккаунт администратора",
      ),
    );
    expect(onClose).toHaveBeenCalled();
  });

  it("409 «last admin account» → toast.warning о последнем админе, onClose", async () => {
    vi.mocked(adminApi.removeUser).mockRejectedValue(
      fakeAxios409("Cannot delete the last admin account"),
    );
    const onClose = vi.fn();
    renderModal({ onClose });

    fireEvent.click(screen.getByRole("button", { name: /^удалить$/i }));

    await waitFor(() =>
      expect(toast.warning).toHaveBeenCalledWith(
        "Нельзя удалить последнего администратора в системе",
      ),
    );
    expect(onClose).toHaveBeenCalled();
  });
});
