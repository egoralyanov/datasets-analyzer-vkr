// Презентационный компонент тела toast'а — вынесен из lib/toast.tsx, чтобы
// lib/toast.tsx экспортировал только helper-функции (требование
// react-refresh/only-export-components).
//
// Стиль scientific/archive: paper-50 фон, hairline border-paper-300 +
// 3px-руль слева в семантическом цвете, иконка lucide, кнопка-крестик
// для ручного закрытия. Углы прямые. Без shadow.
import { AlertTriangle, Check, Info, X } from "lucide-react";
import { toast as hotToast, type Toast as HotToast } from "react-hot-toast";

export type ToastTone = "success" | "error" | "warning" | "info";

const BORDER: Record<ToastTone, string> = {
  success: "border-success-500",
  info: "border-info-500",
  warning: "border-warning-500",
  error: "border-critical-500",
};

const ICON_COLOR: Record<ToastTone, string> = {
  success: "text-success-700",
  info: "text-info-700",
  warning: "text-warning-700",
  error: "text-critical-700",
};

const ICON: Record<ToastTone, typeof Check> = {
  success: Check,
  info: Info,
  warning: AlertTriangle,
  error: AlertTriangle,
};

export function ToastBody({
  text,
  tone,
  t,
}: {
  text: string;
  tone: ToastTone;
  t: HotToast;
}) {
  const Icon = ICON[tone];
  return (
    <div
      className={`pointer-events-auto flex max-w-sm items-start gap-3 border border-paper-300 border-l-[3px] ${BORDER[tone]} bg-paper-50 px-4 py-3 font-sans text-sm text-paper-800`}
      role="status"
      aria-live={tone === "error" || tone === "warning" ? "assertive" : "polite"}
    >
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${ICON_COLOR[tone]}`} />
      <p className="flex-1 leading-relaxed">{text}</p>
      <button
        type="button"
        onClick={() => hotToast.dismiss(t.id)}
        className="shrink-0 text-paper-400 transition-colors hover:text-ink-700"
        aria-label="Закрыть уведомление"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
