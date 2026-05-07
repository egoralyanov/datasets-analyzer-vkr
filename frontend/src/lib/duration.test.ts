// Юнит-тесты computeDuration (Sprint 6, Phase 8.1).
//
// Регрессия: до фикса любой интервал < 500 мс округлялся до 0 секунд и
// выводился как «0 сек» — выглядело как баг (профайлинг Iris занимает
// сотни миллисекунд на warm-cache, не ноль).
import { describe, it, expect } from "vitest";

import { computeDuration } from "./duration";

function isoPlus(startMs: number, addedMs: number): string {
  return new Date(startMs + addedMs).toISOString();
}

describe("computeDuration", () => {
  const startMs = Date.UTC(2026, 4, 7, 12, 0, 0);
  const startIso = new Date(startMs).toISOString();

  it("возвращает null если finishedIso=null", () => {
    expect(computeDuration(startIso, null)).toBeNull();
  });

  it("0.3 сек → «<1 сек»", () => {
    expect(computeDuration(startIso, isoPlus(startMs, 300))).toBe("<1 сек");
  });

  it("0 мс → «<1 сек» (одна и та же метка)", () => {
    expect(computeDuration(startIso, startIso)).toBe("<1 сек");
  });

  it("1.6 сек → «2 сек» (округление до ближайшей)", () => {
    expect(computeDuration(startIso, isoPlus(startMs, 1600))).toBe("2 сек");
  });

  it("15 сек → «15 сек»", () => {
    expect(computeDuration(startIso, isoPlus(startMs, 15_000))).toBe("15 сек");
  });

  it("90 сек → «1 мин 30 сек»", () => {
    expect(computeDuration(startIso, isoPlus(startMs, 90_000))).toBe(
      "1 мин 30 сек",
    );
  });

  it("120 сек → «2 мин» (без хвоста)", () => {
    expect(computeDuration(startIso, isoPlus(startMs, 120_000))).toBe("2 мин");
  });
});
