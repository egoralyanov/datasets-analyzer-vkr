// Render-тесты DeleteAnalysisModal (Sprint 6, Phase 5.2).
//
// Покрытие:
// - Заголовок «Удалить анализ?» + тело с filename, taskTypeLabel, dateLabel.
// - completedAt=Date → "от <дата>", completedAt=null → "запущен <дата>"
//   (дата берётся из startedAt).
// - taskTypeCode=null → скобки с лейблом скрыты, всё остальное в теле есть.
//
// API-вызов analysesApi.remove мокаем через vi.mock.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { DeleteAnalysisModal } from "./DeleteAnalysisModal";

vi.mock("../../api/analyses", () => ({
  analysesApi: {
    remove: vi.fn(),
  },
}));

function renderModal(props: Partial<Parameters<typeof DeleteAnalysisModal>[0]> = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const defaults = {
    open: true,
    onClose: () => {},
    analysisId: "an-1",
    datasetFilename: "iris.csv",
    taskTypeCode: "BINARY_CLASSIFICATION" as const,
    completedAt: "2026-05-07T14:32:00Z",
    startedAt: "2026-05-07T14:30:00Z",
  };
  return render(
    <QueryClientProvider client={client}>
      <DeleteAnalysisModal {...defaults} {...props} />
    </QueryClientProvider>,
  );
}

describe("DeleteAnalysisModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("рендерит заголовок «Удалить анализ?» и тело с filename + taskTypeLabel", () => {
    renderModal();
    expect(
      screen.getByRole("heading", { name: "Удалить анализ?" }),
    ).toBeInTheDocument();
    // Тело лежит в одном <p>; проверяем по textContent, чтобы не зависеть от
    // вложенных <span> для лейбла типа задачи.
    const body = screen.getByText(/анализ «iris\.csv»/i);
    expect(body.textContent).toContain("iris.csv");
    expect(body.textContent).toContain("Бинарная классификация");
    expect(body.textContent).toContain("PDF-отчёт также будет удалён");
    expect(body.textContent).toContain("Действие необратимо");
  });

  it("completedAt=Date → подпись «от …»", () => {
    renderModal({
      completedAt: new Date("2026-05-07T14:32:00Z"),
    });
    const body = screen.getByText(/анализ «iris\.csv»/i);
    expect(body.textContent).toMatch(/, от /);
    expect(body.textContent).not.toMatch(/, запущен /);
  });

  it("completedAt=null → подпись «запущен …» (берётся из startedAt)", () => {
    renderModal({
      completedAt: null,
      startedAt: "2026-05-07T10:00:00Z",
    });
    const body = screen.getByText(/анализ «iris\.csv»/i);
    expect(body.textContent).toMatch(/, запущен /);
    expect(body.textContent).not.toMatch(/, от /);
  });

  it("taskTypeCode=null → скобки с лейблом скрыты", () => {
    renderModal({ taskTypeCode: null });
    const body = screen.getByText(/анализ «iris\.csv»/i);
    // Без скобок не должно быть лейбла, но дата и хвост на месте.
    expect(body.textContent).not.toContain("Бинарная классификация");
    expect(body.textContent).toMatch(/, от /);
    expect(body.textContent).toContain("PDF-отчёт также будет удалён");
  });
});
