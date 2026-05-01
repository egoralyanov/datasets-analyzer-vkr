// Зеркальный тип ответа /api/health (см. backend/app/api/health.py).
export interface HealthResponse {
  status: string;
  service: string;
}
