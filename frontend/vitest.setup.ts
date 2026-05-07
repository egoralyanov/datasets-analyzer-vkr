// Подключение @testing-library/jest-dom для расширенных matcher'ов
// (toBeInTheDocument, toHaveTextContent и т.д.) в render-тестах.
import "@testing-library/jest-dom/vitest";

// Авто-очистка DOM между тестами. Без этого render() из предыдущего теста
// оставляет узлы в document.body, и screen.getBy* находит несколько
// совпадений (см. https://testing-library.com/docs/react-testing-library/api/#cleanup).
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
