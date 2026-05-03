# Makefile для проекта «Анализатор»
# Все команды разработки идут через docker compose v2.

.PHONY: up down logs migrate migrate-create seed seed-rules seed-catalog seed-all build-real-set build-synthetic-set train-meta test clean

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

# === ML pipeline (Спринт 3) ===

# Сборка реальной части обучающей выборки + каталога: 30 датасетов из sklearn/UCI/GitHub.
# Выход: backend/ml/data/real_set.json. Идемпотентно по (title, n_rows, n_cols).
build-real-set:
	docker compose exec backend python -m ml.build_real_set

# Сборка синтетической части обучающей выборки: 75 + 75 = 150 датасетов
# через make_classification и make_regression. Выход: backend/ml/data/synthetic_set.json.
build-synthetic-set:
	docker compose exec backend python -m ml.build_synthetic_set

# Обучение мета-классификатора (Random Forest) на real + synthetic выборках.
# Выход: backend/ml/models/{scaler.pkl, meta_classifier.pkl, meta_classifier_report.json}.
train-meta:
	docker compose exec backend python -m ml.train_meta_classifier

# Заливка каталога external_datasets в БД из real_set.json + scaler.pkl.
# Идемпотентно через ON CONFLICT (title) DO UPDATE; embedding'и пересчитываются
# при изменении scaler'а или meta_features.
seed-catalog:
	docker compose exec backend python -m seeds.seed_external_datasets

# Полный сид: правила качества + мета-классификатор (если нужен) + каталог.
# Порядок важен: seed-catalog требует обученного scaler.pkl (train-meta),
# и наличия таблицы external_datasets (миграция Phase 1 уже применена через `make migrate`).
seed-all: seed-rules
	@if [ ! -f backend/ml/data/real_set.json ]; then \
		echo "real_set.json не найден — запустите 'make build-real-set'"; exit 1; \
	fi
	@if [ ! -f backend/ml/data/synthetic_set.json ]; then \
		echo "synthetic_set.json не найден — запустите 'make build-synthetic-set'"; exit 1; \
	fi
	@if [ ! -f backend/ml/models/meta_classifier.pkl ]; then \
		$(MAKE) train-meta; \
	fi
	$(MAKE) seed-catalog

# Запустить unit-тесты бэкенда внутри контейнера
test:
	docker compose exec backend pytest

# Полная очистка: остановить контейнеры и удалить volumes (БД будет очищена)
clean:
	docker compose down -v
