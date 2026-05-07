import { describe, it, expect } from "vitest";
import { pluralize } from "./pluralize";

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
