// Тонкая обёртка над react-hot-toast в scientific/archive-стиле.
//
// Архив-эстетика (см. frontend/DESIGN_TOKENS.md, разделы 5-6) запрещает:
// rounded-md/lg, shadow-md, gradient, fill-кнопки. Поэтому дефолтные стили
// react-hot-toast не подходят, и каждый toast рендерим через `toast.custom`
// с собственной разметкой. Презентационный компонент ToastBody вынесен в
// components/ui/ToastBody.tsx (Fast Refresh требует, чтобы файл экспортировал
// ИЛИ компоненты, ИЛИ значения, но не то и другое одновременно).
//
// Длительность: info/success — 4с (короткое подтверждение),
// warning/error — 6с (нужно успеть прочитать «что пошло не так»).
//
// Точка вызова единая: `import { toast } from "@/lib/toast"`.
// `toast.success("...") | toast.error("...") | toast.warning("...") | toast.info("...")`.
import { toast as hotToast } from "react-hot-toast";

import { ToastBody, type ToastTone } from "../components/ui/ToastBody";

const DURATIONS: Record<ToastTone, number> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

function emit(text: string, tone: ToastTone) {
  hotToast.custom(
    (t) => <ToastBody text={text} tone={tone} t={t} />,
    { duration: DURATIONS[tone] },
  );
}

export const toast = {
  success: (text: string) => emit(text, "success"),
  error: (text: string) => emit(text, "error"),
  warning: (text: string) => emit(text, "warning"),
  info: (text: string) => emit(text, "info"),
};
