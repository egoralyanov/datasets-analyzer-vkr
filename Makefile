# Makefile для проекта «Анализатор»
# Все команды разработки идут через docker compose v2.

.PHONY: up down logs migrate migrate-create seed test clean

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

# Загрузка справочников (заглушка для Спринта 0, реальная реализация в Спринте 2+)
seed:
	@echo "seed: заглушка — реальная загрузка справочников появится в Спринте 2+"

# Запустить unit-тесты бэкенда внутри контейнера
test:
	docker compose exec backend pytest

# Полная очистка: остановить контейнеры и удалить volumes (БД будет очищена)
clean:
	docker compose down -v
