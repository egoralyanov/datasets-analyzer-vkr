// Toaster — обёртка над react-hot-toast Toaster, монтируется один раз в App.
//
// Расположение: bottom-right. Смещение по высоте 64px — над ServerStatus-
// индикатором, который пиннится в `fixed bottom-4 right-4`. Иначе toast
// перекрывал бы его.
//
// Helper-методы для эмита уведомлений — в lib/toast.tsx (`toast.success/...`).
// Разделение по файлам нужно из-за react-refresh/only-export-components:
// Fast Refresh не любит файл, который экспортирует и компонент, и
// произвольные значения.
import { Toaster as HotToaster } from "react-hot-toast";

export function Toaster() {
  return (
    <HotToaster
      position="bottom-right"
      gutter={8}
      containerStyle={{ inset: "auto 16px 64px auto" }}
    />
  );
}
