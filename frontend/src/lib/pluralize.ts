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

export function pluralize(
  n: number,
  forms: readonly [one: string, few: string, many: string],
): string {
  const abs = Math.abs(Math.trunc(n));
  const mod100 = abs % 100;
  const mod10 = abs % 10;

  if (mod100 >= 11 && mod100 <= 14) return forms[2];
  if (mod10 === 1) return forms[0];
  if (mod10 >= 2 && mod10 <= 4) return forms[1];
  return forms[2];
}
