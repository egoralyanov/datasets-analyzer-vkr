// Утилиты форматирования с локалью ru-RU.

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toLocaleString("ru-RU", { maximumFractionDigits: 1 })} КБ`;
  const mb = kb / 1024;
  return `${mb.toLocaleString("ru-RU", { maximumFractionDigits: 1 })} МБ`;
}

export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatNumber(n: number): string {
  return n.toLocaleString("ru-RU");
}
