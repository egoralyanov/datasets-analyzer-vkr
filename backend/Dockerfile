FROM python:3.11-slim

# Системные пакеты для WeasyPrint (PDF-отчёты в Спринте 5).
# Ставим заранее, чтобы при подключении weasyprint не пересобирать базовый образ.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    fonts-dejavu \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем манифест и все пакеты приложения. setuptools на момент editable install
# должен видеть и app, и seeds — иначе модули не попадут в site-packages.
COPY pyproject.toml ./
COPY app ./app
COPY seeds ./seeds

# Устанавливаем зависимости вместе с dev-инструментами (pytest, httpx — нужны для make test).
# Editable-install (-e): site-packages указывает на /app, поэтому при bind-mount
# из docker-compose pytest и uvicorn видят актуальный код без пересборки образа.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Конфигурация и шаблоны миграций Alembic.
COPY alembic.ini ./
COPY alembic ./alembic

# Тесты — нужны для `make test` (docker compose exec backend pytest).
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
