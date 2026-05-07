import { describe, it, expect } from "vitest";
import { PLURAL_FORMS, pluralize } from "./pluralize";

describe("pluralize — русские числительные", () => {
  const FORMS = ["замечание", "замечания", "замечаний"] as const;

  it("1 → одушевлённая форма (one)", () => {
    expect(pluralize(1, FORMS)).toBe("замечание");
  });

  it("2 → форма 2-4 (few)", () => {
    expect(pluralize(2, FORMS)).toBe("замечания");
  });

  it("4 → форма 2-4 (few)", () => {
    expect(pluralize(4, FORMS)).toBe("замечания");
  });

  it("5 → форма 5+ (many)", () => {
    expect(pluralize(5, FORMS)).toBe("замечаний");
  });

  it("11 → форма many (исключение mod100 ∈ 11..14)", () => {
    expect(pluralize(11, FORMS)).toBe("замечаний");
  });

  it("21 → форма one (mod10 == 1, mod100 != 11)", () => {
    expect(pluralize(21, FORMS)).toBe("замечание");
  });

  it("22 → форма few (mod10 == 2, mod100 != 12)", () => {
    expect(pluralize(22, FORMS)).toBe("замечания");
  });
});

describe("PLURAL_FORMS — именованные tuples (Sprint 6, Phase 5.1)", () => {
  it.each([
    [1, "анализ"],
    [2, "анализа"],
    [5, "анализов"],
  ])("analysis: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.analysis)).toBe(expected);
  });

  it.each([
    [1, "отчёт"],
    [2, "отчёта"],
    [5, "отчётов"],
  ])("report: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.report)).toBe(expected);
  });

  it.each([
    [1, "PDF-отчёт"],
    [2, "PDF-отчёта"],
    [5, "PDF-отчётов"],
  ])("pdfReport: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.pdfReport)).toBe(expected);
  });

  it.each([
    [1, "датасет"],
    [2, "датасета"],
    [5, "датасетов"],
  ])("dataset: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.dataset)).toBe(expected);
  });

  it.each([
    [1, "запись"],
    [2, "записи"],
    [5, "записей"],
  ])("record: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.record)).toBe(expected);
  });

  it.each([
    [1, "предупреждение"],
    [2, "предупреждения"],
    [5, "предупреждений"],
  ])("warning: %i → %s", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.warning)).toBe(expected);
  });

  it.each([
    [1, "критичное"],
    [2, "критичных"],
    [5, "критичных"],
  ])("critical: %i → %s (few/many совпадают)", (n, expected) => {
    expect(pluralize(n, PLURAL_FORMS.critical)).toBe(expected);
  });
});
