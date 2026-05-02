// HTTP-клиент для общения с бэкендом.
// baseURL '/api' — все запросы уходят через nginx-прокси (см. frontend/nginx.conf),
// поэтому одинаков для dev и production: единый origin, без CORS.
import axios from "axios";
import { useAuthStore } from "../store/authStore";

export const apiClient = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// Подставляем Bearer-токен из authStore в каждый запрос, если он есть.
apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// На 401 (просроченный/невалидный токен) и 403 (нет токена для защищённого эндпоинта)
// чистим локальный auth-state и отправляем пользователя на /login.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    if (status === 401 || status === 403) {
      useAuthStore.getState().logout();
      // Не редиректим, если пользователь уже на /login или /register —
      // иначе при ошибке логина страница перезагрузится и сотрёт сообщение.
      const path = window.location.pathname;
      if (path !== "/login" && path !== "/register") {
        window.location.href = "/login";
      }
    }
    return Promise.reject(error);
  },
);
