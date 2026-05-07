// Vitest-конфиг (Спринт 6, Phase 5).
//
// jsdom — для render-тестов компонентов через @testing-library/react.
// Setup-файл импортирует custom matchers из @testing-library/jest-dom.
//
// Vitest сам читает Vite-конфиг (плагины, optimizeDeps), поэтому здесь
// только специфичные для test-окружения настройки.
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: false,
  },
});
