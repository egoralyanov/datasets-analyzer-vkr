"""SQLAlchemy: engine, sessionmaker, базовый класс для ORM-моделей."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей приложения."""


# pool_pre_ping проверяет соединение перед использованием — защита от «протухших» коннектов.
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
