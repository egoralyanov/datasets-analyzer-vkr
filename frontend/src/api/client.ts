// HTTP-клиент для общения с бэкендом.
// baseURL '/api' — все запросы уходят через nginx-прокси (см. frontend/nginx.conf),
// поэтому одинаков для dev и production: единый origin, без CORS.
// Авторизационные interceptor'ы (Bearer-токен, обработка 401) появятся в Спринте 1.
import axios from "axios";

export const apiClient = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 10_000,
});
