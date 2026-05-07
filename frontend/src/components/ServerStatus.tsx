// Индикатор связи фронта с бэкендом. Опрашивает /api/health через React Query
// каждые 30 секунд.
//
// Стиль (Sprint 5, Phase 3.2): негативный индикатор. В нормальном состоянии
// (сервер отвечает или идёт первая проверка) ничего не показываем — UI и так
// ведёт себя корректно. Плашка появляется ТОЛЬКО при потере связи: приглушённая
// critical-плашка в архивном стиле, фиксирована в правом нижнем углу.
import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../api/health";

export function ServerStatus() {
  const { data, isError, isPending } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
  });

  // Pending первой проверки и нормальная связь — не показываем ничего.
  if (isPending) return null;
  if (!isError && data?.status === "ok") return null;

  return (
    <div className="border-l-[3px] border-critical-500 bg-paper-50 px-4 py-2 shadow-overlay">
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-critical-700">
        ⚠ СВЯЗЬ С СЕРВЕРОМ ПОТЕРЯНА
      </p>
      <p className="mt-1 font-serif text-xs leading-relaxed text-paper-600">
        Проверьте, что бэкенд запущен.
      </p>
    </div>
  );
}
