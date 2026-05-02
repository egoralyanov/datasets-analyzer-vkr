// Лимит размера загружаемого файла на фронте — UX-страховка перед отправкой.
// На бэке такой же лимит проверяется через settings.MAX_FILE_SIZE_MB
// (см. backend/app/api/datasets.py). Две границы: фронт для UX, бэк для безопасности.
export const MAX_FILE_SIZE_MB = 100;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export const ALLOWED_DATASET_EXTENSIONS = ["csv", "xlsx"] as const;
