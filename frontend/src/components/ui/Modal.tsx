// Базовая модалка в scientific/archive-стиле (Sprint 6, Phase 5).
//
// Стилистика повторяет StartAnalysisModal: paper-900/50 overlay, paper-50 фон,
// hairline border-paper-300, единственная разрешённая `shadow-overlay`,
// Plex Serif для заголовка + Plex Sans uppercase tracking-wider для
// подзаголовка-«рубрики». Углы прямые (`rounded-none`) — DESIGN_TOKENS.md, п.4.
//
// Фичи:
// - Esc и click outside закрывают модалку (если `disableDismiss=false`).
// - `disableDismiss` — для активной мутации: блокирует backdrop-click, Esc
//   и крестик. Кнопки внутри модалки сами решают свою disabled-логику.
// - Focus-trap: при mount фиксируется `document.activeElement` и автофокус
//   уходит на первый focusable внутри модалки. Tab/Shift+Tab закольцовывают
//   фокус по interactive-элементам. При unmount фокус возвращается на
//   сохранённый элемент.
// - role="dialog" + aria-modal + aria-labelledby для скрин-ридеров.
//
// StartAnalysisModal специально не рефакторим под этот компонент —
// рефакторинг отдельной задачей в Phase 8 (риск регрессии не оправдан).
import { useEffect, useId, useRef } from "react";

const SIZE_CLASSES = {
  md: "max-w-md",
  lg: "max-w-lg",
} as const;

type Size = keyof typeof SIZE_CLASSES;

type Props = {
  open: boolean;
  onClose: () => void;
  title: string;
  /** Маленький лейбл uppercase tracking-wider над title (типа «УДАЛЕНИЕ»). */
  subtitle?: string;
  /** Если true — esc, click-outside и крестик не закрывают модалку. */
  disableDismiss?: boolean;
  /** Контент модалки (тело + кнопки). Пробрасывается как children. */
  children: React.ReactNode;
  size?: Size;
};

export function Modal({
  open,
  onClose,
  title,
  subtitle,
  disableDismiss = false,
  children,
  size = "md",
}: Props) {
  if (!open) return null;
  return (
    <ModalInner
      onClose={onClose}
      title={title}
      subtitle={subtitle}
      disableDismiss={disableDismiss}
      size={size}
    >
      {children}
    </ModalInner>
  );
}

type InnerProps = Omit<Props, "open">;

function ModalInner({
  onClose,
  title,
  subtitle,
  disableDismiss,
  children,
  size = "md",
}: InnerProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  // Esc + focus-trap. Возврат фокуса на предыдущий activeElement.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;

    // Автофокус на первый focusable. Откладываем на следующий tick, чтобы
    // dialog был уже примонтирован и querySelectorAll нашёл узлы.
    const t = setTimeout(() => {
      const first = getFocusables(dialogRef.current)[0];
      first?.focus();
    }, 0);

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !disableDismiss) {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key === "Tab") {
        const focusables = getFocusables(dialogRef.current);
        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey) {
          if (active === first || !dialogRef.current?.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      clearTimeout(t);
      previouslyFocused?.focus?.();
    };
  }, [disableDismiss, onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-paper-900/50 p-4"
      onClick={() => !disableDismiss && onClose()}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`w-full ${SIZE_CLASSES[size]} border border-paper-300 bg-paper-50 p-6 shadow-overlay`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            {subtitle && (
              <p className="font-sans text-xs font-medium uppercase tracking-wider text-paper-500">
                {subtitle}
              </p>
            )}
            <h2
              id={titleId}
              className="mt-1 font-serif text-[1.375rem] font-semibold leading-snug text-paper-900"
            >
              {title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={disableDismiss}
            className="font-sans text-sm text-paper-500 transition-colors hover:text-ink-700 disabled:opacity-50"
            aria-label="Закрыть"
          >
            ✕
          </button>
        </div>
        <div className="mt-4">{children}</div>
      </div>
    </div>
  );
}

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(", ");

function getFocusables(root: HTMLElement | null): HTMLElement[] {
  if (!root) return [];
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (el) => el.offsetParent !== null || el === document.activeElement,
  );
}
