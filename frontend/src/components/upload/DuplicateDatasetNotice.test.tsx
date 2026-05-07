// Render-тесты DuplicateDatasetNotice (Sprint 6, Phase 5.3).
//
// Покрытие:
// - Заголовок и подтекст блока на месте (текст-маркер).
// - Клик «ВЫБРАТЬ ДРУГОЙ ФАЙЛ» вызывает onChooseAnother без аргументов.
// - Клик «ОТКРЫТЬ СУЩЕСТВУЮЩИЙ» вызывает onOpenExisting с правильным id.
// - При смене existingDatasetId блок берёт новый id (вторая попытка дубля).
// - isOpening=true дизейблит обе кнопки.
//
// Используем fireEvent из @testing-library/react (встроенный) — пакет
// @testing-library/user-event в проекте не подключён, и тащить его ради
// одного click'а не стали (Phase 5.3 — без новых зависимостей).
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

import { DuplicateDatasetNotice } from "./DuplicateDatasetNotice";

function setup(props: Partial<Parameters<typeof DuplicateDatasetNotice>[0]> = {}) {
  const defaults = {
    existingDatasetId: "ds-1",
    isOpening: false,
    onOpenExisting: vi.fn(),
    onChooseAnother: vi.fn(),
  };
  const merged = { ...defaults, ...props };
  return {
    ...merged,
    ...render(<DuplicateDatasetNotice {...merged} />),
  };
}

describe("DuplicateDatasetNotice", () => {
  it("рендерит заголовок и подтекст", () => {
    setup();
    expect(
      screen.getByRole("heading", { name: "Этот файл уже загружен" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/в вашей библиотеке есть датасет с идентичным содержимым/i),
    ).toBeInTheDocument();
  });

  it("клик «ВЫБРАТЬ ДРУГОЙ ФАЙЛ» вызывает onChooseAnother", () => {
    const onChooseAnother = vi.fn();
    setup({ onChooseAnother });
    fireEvent.click(
      screen.getByRole("button", { name: /выбрать другой файл/i }),
    );
    expect(onChooseAnother).toHaveBeenCalledTimes(1);
    expect(onChooseAnother).toHaveBeenCalledWith();
  });

  it("клик «ОТКРЫТЬ СУЩЕСТВУЮЩИЙ» вызывает onOpenExisting с правильным id", () => {
    const onOpenExisting = vi.fn();
    setup({ existingDatasetId: "ds-42", onOpenExisting });
    fireEvent.click(
      screen.getByRole("button", { name: /открыть существующий/i }),
    );
    expect(onOpenExisting).toHaveBeenCalledTimes(1);
    expect(onOpenExisting).toHaveBeenCalledWith("ds-42");
  });

  it("при rerender с другим existingDatasetId передаёт новый id", () => {
    const onOpenExisting = vi.fn();
    const { rerender } = render(
      <DuplicateDatasetNotice
        existingDatasetId="ds-first"
        isOpening={false}
        onOpenExisting={onOpenExisting}
        onChooseAnother={() => {}}
      />,
    );
    rerender(
      <DuplicateDatasetNotice
        existingDatasetId="ds-second"
        isOpening={false}
        onOpenExisting={onOpenExisting}
        onChooseAnother={() => {}}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /открыть существующий/i }),
    );
    expect(onOpenExisting).toHaveBeenCalledWith("ds-second");
  });

  it("isOpening=true дизейблит обе кнопки", () => {
    setup({ isOpening: true });
    expect(
      screen.getByRole("button", { name: /открыть существующий/i }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: /выбрать другой файл/i }),
    ).toBeDisabled();
  });
});
