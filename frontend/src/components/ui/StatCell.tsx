// Stat-карточка scientific/archive: лейбл uppercase tracking-wider сверху,
// крупная цифра в Plex Mono снизу, опциональный hint-подтекст.
// Sprint 6, Phase 7: используется на dashboard для блока «Ваша статистика».
//
// На /admin живёт локальный StatCard почти-идентичной структуры (см.
// pages/Admin.tsx). Phase 7 не трогает админку — рефакторинг на общий
// компонент откладывается до следующей задачи.
type Props = {
  label: string;
  value: number;
  /** Опциональный мелкий подтекст (например «из 12») под цифрой. */
  hint?: string;
};

export function StatCell({ label, value, hint }: Props) {
  return (
    <div className="bg-paper-50 px-5 py-5">
      <p className="font-sans text-[0.6875rem] font-medium uppercase tracking-wider text-paper-500">
        {label.toUpperCase()}
      </p>
      <p className="mt-2 font-mono text-[2rem] font-medium leading-none text-paper-900">
        {value.toLocaleString("ru-RU")}
      </p>
      {hint && (
        <p className="mt-1.5 font-sans text-xs text-paper-500">{hint}</p>
      )}
    </div>
  );
}
