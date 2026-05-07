// Стандартное русское склонение по числу.
//
// Правила (см. https://learn.javascript.ru/intl-pluralrules + раздел «Числовые
// формы» в АН СССР, 1980):
//   - mod100 ∈ [11..14]                        → many («11 замечаний», «113 замечаний»)
//   - mod10 == 1                               → one  («1 замечание», «21 замечание»)
//   - mod10 ∈ [2..4]                           → few  («2 замечания», «23 замечания»)
//   - иначе                                    → many («5 замечаний», «27 замечаний»)
//
// Forms — кортеж [one, few, many]. Если все три формы совпадают (например, для
// неизменяемого «инфо»), просто передайте одно слово трижды или используйте
// функцию-обёртку — но обычно так делать не нужно.

export type PluralForms = readonly [one: string, few: string, many: string];

export function pluralize(n: number, forms: PluralForms): string {
  const abs = Math.abs(Math.trunc(n));
  const mod100 = abs % 100;
  const mod10 = abs % 10;

  if (mod100 >= 11 && mod100 <= 14) return forms[2];
  if (mod10 === 1) return forms[0];
  if (mod10 >= 2 && mod10 <= 4) return forms[1];
  return forms[2];
}

// Именованные формы для типовых сущностей системы. Используются вместо
// inline-литералов, чтобы при изменении формулировки (например, «PDF-отчёт» →
// «PDF-документ») точка правки была одна. Sprint 6, Phase 5.1.
export const PLURAL_FORMS = {
  analysis: ["анализ", "анализа", "анализов"] as PluralForms,
  report: ["отчёт", "отчёта", "отчётов"] as PluralForms,
  pdfReport: ["PDF-отчёт", "PDF-отчёта", "PDF-отчётов"] as PluralForms,
  dataset: ["датасет", "датасета", "датасетов"] as PluralForms,
  record: ["запись", "записи", "записей"] as PluralForms,
  // Сводка качества (см. components/analysis/QualityFlags.tsx).
  remark: ["замечание", "замечания", "замечаний"] as PluralForms,
  warning: [
    "предупреждение",
    "предупреждения",
    "предупреждений",
  ] as PluralForms,
  // critical: ед. число «КРИТИЧНОЕ», few/many совпадают «КРИТИЧНЫХ»; форма
  // подаётся в нижнем регистре, а uppercase делает CSS на стороне UI.
  critical: ["критичное", "критичных", "критичных"] as PluralForms,
} as const;
