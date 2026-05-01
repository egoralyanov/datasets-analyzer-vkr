# Стадия 1: сборка React-приложения с помощью Vite.
FROM node:20-alpine AS builder
WORKDIR /app

# Сначала только манифесты — для кэширования слоя установки зависимостей.
COPY package.json package-lock.json ./
RUN npm ci

# Затем исходники и сборка production-бандла.
COPY . .
RUN npm run build

# Стадия 2: раздача статики через Nginx + проксирование /api/ к backend:8000.
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
