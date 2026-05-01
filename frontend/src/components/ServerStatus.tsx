import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../api/health";

// Индикатор связи фронта с бэкендом. Опрашивает /api/health через React Query
// каждые 30 секунд; зелёная точка — сервер отвечает, красная — нет, серая — идёт первая проверка.
export function ServerStatus() {
  const { data, isError, isPending } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
  });

  if (isPending) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="h-2 w-2 rounded-full bg-gray-400 animate-pulse" />
        Проверка соединения…
      </div>
    );
  }

  if (isError || data?.status !== "ok") {
    return (
      <div className="flex items-center gap-2 text-sm text-red-600">
        <span className="h-2 w-2 rounded-full bg-red-500" />
        Сервер не отвечает
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm text-green-700">
      <span className="h-2 w-2 rounded-full bg-green-500" />
      Сервер на связи
    </div>
  );
}
