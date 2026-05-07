// Sprint 6, Phase 5.2: маленький тест шаред-словаря TASK_TYPE_LABEL и
// helper'а getTaskTypeLabel. Проверяем все 5 валидных кодов и 3 фолбэка.
import { describe, it, expect } from "vitest";

import { TASK_TYPE_LABEL, getTaskTypeLabel } from "./taskTypes";

describe("TASK_TYPE_LABEL", () => {
  it("мапит все 5 валидных кодов", () => {
    expect(TASK_TYPE_LABEL.BINARY_CLASSIFICATION).toBe("Бинарная классификация");
    expect(TASK_TYPE_LABEL.MULTICLASS_CLASSIFICATION).toBe(
      "Многоклассовая классификация",
    );
    expect(TASK_TYPE_LABEL.REGRESSION).toBe("Регрессия");
    expect(TASK_TYPE_LABEL.CLUSTERING).toBe("Кластеризация");
    expect(TASK_TYPE_LABEL.NOT_READY).toBe("Данные не готовы для ML");
  });
});

describe("getTaskTypeLabel", () => {
  it("возвращает подпись для известного кода", () => {
    expect(getTaskTypeLabel("REGRESSION")).toBe("Регрессия");
  });

  it("возвращает null для null/undefined/пустой строки", () => {
    expect(getTaskTypeLabel(null)).toBeNull();
    expect(getTaskTypeLabel(undefined)).toBeNull();
    expect(getTaskTypeLabel("")).toBeNull();
  });

  it("возвращает null для неизвестного кода", () => {
    expect(getTaskTypeLabel("SOMETHING_WEIRD")).toBeNull();
  });
});
