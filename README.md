# Интеллектуальная система анализа наборов данных для решения задач машинного обучения

Выпускная квалификационная работа бакалавра

## Описание

Веб-приложение для первичной диагностики табличных датасетов перед задачами машинного обучения. Загружает CSV/XLSX, выполняет EDA-профилирование, выявляет проблемы качества данных и формирует объяснимые рекомендации по выбору типа ML-задачи (классификация, регрессия, кластеризация).

## Стек

- **Backend:** Python 3.11, FastAPI, SQLAlchemy 2, Alembic, Pydantic 2
- **Frontend:** React 18 + TypeScript, Vite, Tailwind CSS
- **Database:** PostgreSQL 15 + pgvector
- **DevOps:** Docker Compose, Nginx

## Требования

- Docker 24+ и Docker Compose v2
- GNU Make
- Свободные порты: `3000` (фронт)

## Быстрый старт

```bash
# 1. Создать .env из шаблона и подставить криптографический JWT_SECRET
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# (полученное значение записать в .env как JWT_SECRET)

# 2. Поднять все сервисы
make up

# 3. Применить миграции БД
make migrate

# 4. Открыть приложение
# http://localhost:3000
```

## Команды Makefile

| Команда | Назначение |
|---|---|
| `make up` | Поднять все контейнеры |
| `make down` | Остановить контейнеры |
| `make logs` | Посмотреть логи |
| `make migrate` | Применить миграции БД |
| `make seed` | Загрузить справочники (заглушка в Спринте 0) |
| `make test` | Запустить тесты бэкенда |
| `make clean` | Полная очистка (удаление volumes) |

## Структура репозитория

```
.
├── backend/            Python + FastAPI
├── frontend/           React + TypeScript
├── data/               Локальное хранилище (датасеты, отчёты)
├── docker-compose.yml
├── Makefile
└── .env.example
```
