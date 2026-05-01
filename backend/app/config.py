"""
Конфигурация приложения через Pydantic Settings.

Все переменные читаются из переменных окружения (в docker-compose инжектятся
через env_file из корневого .env). Локально, при запуске вне контейнера,
дополнительно подхватывается ./.env.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из окружения / .env-файла."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Игнорируем переменные, не описанные в схеме (POSTGRES_USER и т.д.).
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Анализатор"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str

    # JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 часа

    # File storage
    DATASETS_DIR: str = "/data/datasets"
    REPORTS_DIR: str = "/data/reports"
    MAX_FILE_SIZE_MB: int = 100

    # Analysis
    SAMPLING_THRESHOLD_ROWS: int = 50000
    BACKGROUND_TASK_TIMEOUT_S: int = 120


settings = Settings()
