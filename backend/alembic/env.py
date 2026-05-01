"""
Конфигурация окружения Alembic.

URL подключения к БД берётся не из alembic.ini, а из настроек приложения
(app.config.settings.DATABASE_URL) — единый источник правды. target_metadata
указывает на Base из app.core.db и используется при autogenerate.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.core.db import Base

# Импорт моделей нужен, чтобы Base.metadata знал обо всех таблицах при autogenerate.
# В Спринте 0 моделей ещё нет — импорт активируется в Спринте 1.
# import app.models  # noqa: F401

config = context.config

# Подменяем sqlalchemy.url из ini-файла на актуальный из настроек приложения.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Запуск миграций без подключения к БД (генерация SQL-скрипта)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запуск миграций с реальным подключением к БД."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
