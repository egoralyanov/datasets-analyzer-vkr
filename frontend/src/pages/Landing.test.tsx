// Smoke-тест гостевой главной (Sprint 6, Phase 6).
//
// Не покрываем каждую секцию §1-§7 — это переусложнение. Достаточно
// проверить, что страница в гостевом режиме рендерится без ошибок и
// содержит ключевые маркеры: hero, заголовки витрины, CTA.
//
// Для гостевого пути zustand-store оставляем по умолчанию (user=null),
// и `useAuthStore` возвращает null → ветка GuestLanding.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, beforeEach } from "vitest";

import { Landing } from "./Landing";
import { useAuthStore } from "../store/authStore";

function renderGuest() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Landing />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Landing (guest)", () => {
  beforeEach(() => {
    // Сбрасываем zustand-стор: гостевая ветка ожидает user=null.
    useAuthStore.setState({ user: null, token: null });
  });

  it("рендерит hero с заголовком АНАЛИЗАТОР", () => {
    renderGuest();
    expect(
      screen.getByRole("heading", { level: 1, name: /анализатор/i }),
    ).toBeInTheDocument();
  });

  it("hero-кнопки ведут на /register и /login", () => {
    renderGuest();
    // На странице две пары CTA-кнопок (hero + финал). Берём всё.
    const registers = screen.getAllByRole("link", {
      name: /создать аккаунт/i,
    });
    const logins = screen.getAllByRole("link", { name: /^войти$/i });
    expect(registers.length).toBeGreaterThanOrEqual(2);
    expect(logins.length).toBeGreaterThanOrEqual(2);
    for (const link of registers) {
      expect(link).toHaveAttribute("href", "/register");
    }
    for (const link of logins) {
      expect(link).toHaveAttribute("href", "/login");
    }
  });

  it("витрина рендерит §1–§7 (по заголовкам секций)", () => {
    renderGuest();
    expect(
      screen.getByRole("heading", { name: /как это работает/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /рекомендация типа задачи/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /сводка датасета/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /качество данных/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /распределения/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /похожие датасеты/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /базовая модель/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /pdf-отчёт/i }),
    ).toBeInTheDocument();
  });

  it("финальный CTA рендерится с заголовком и кнопками", () => {
    renderGuest();
    expect(
      screen.getByRole("heading", { name: /готовы попробовать/i }),
    ).toBeInTheDocument();
  });
});
