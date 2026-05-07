"""Pydantic-схемы для пользователя (запрос/ответ API)."""
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    username: str
    role: str
    created_at: datetime


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    username: str | None = Field(default=None, min_length=3, max_length=100)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class UserStats(BaseModel):
    """
    Личные счётчики пользователя для дашборда (Sprint 6, Phase 7).
    Возвращается из GET /api/me/stats.
    """

    datasets_count: int
    analyses_count: int
    successful_analyses_count: int
    reports_count: int
