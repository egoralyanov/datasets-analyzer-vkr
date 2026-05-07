// Демо-фикстура: реальный анализ датасета Iris, прогнанный через API
// (Sprint 6, Phase 6). UUID и даты заменены на стабильные демо-значения,
// чтобы фронт не показывал случайные строки на гостевой странице.
//
// Использование: импортируется из Landing.tsx (гостевая ветка) и
// прокидывается в существующие компоненты §1-§4 страницы анализа
// (TaskRecommendationCard, DatasetSummary, QualityFlags, Distributions).
//
// Источник: см. plans/06-...md, Phase 6, раздел "Подготовка фикстуры".
import datasetJson from "./dataset.json";
import analysisJson from "./analysis.json";
import resultJson from "./result.json";

import type { Analysis, AnalysisResult } from "../../types/analysis";
import type { DatasetWithPreview } from "../../types/dataset";

// Тип-каст с as unknown — фикстура была сгенерирована реальным API,
// поля совпадают с TS-схемами (см. backend/app/schemas/*.py); отдельно
// валидировать в рантайме не нужно.
export const irisDataset = datasetJson as unknown as DatasetWithPreview;
export const irisAnalysis = analysisJson as unknown as Analysis;
export const irisResult = resultJson as unknown as AnalysisResult;
