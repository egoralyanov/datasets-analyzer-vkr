# Makefile для проекта «Анализатор»
# Все команды разработки идут через docker compose v2.

.PHONY: up down logs migrate migrate-create seed seed-rules test clean

# Поднять все контейнеры (postgres, backend, frontend) в фоне
up:
	docker compose up -d --build

# Остановить контейнеры
down:
	docker compose down

# Логи всех сервисов в реальном времени
logs:
	docker compose logs -f

# Применить миграции БД
migrate:
	docker compose exec backend alembic upgrade head

# Сгенерировать новую миграцию по изменениям в моделях. Использование: make migrate-create name=add_users_table
migrate-create:
	docker compose exec backend alembic revision --autogenerate -m "$(name)"

# Загрузка всех справочников. На Спринте 2 — только quality_rules.
seed: seed-rules

# Загрузка справочника правил качества (12 правил из .knowledge/methods/quality-checks.md).
# Идемпотентно: повторный запуск не плодит дубликаты и не перезаписывает существующие записи.
seed-rules:
	docker compose exec backend python -m seeds.seed_quality_rules

# Запустить unit-тесты бэкенда внутри контейнера
test:
	docker compose exec backend pytest

# Полная очистка: остановить контейнеры и удалить volumes (БД будет очищена)
clean:
	docker compose down -v
