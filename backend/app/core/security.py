"""
Утилиты безопасности: хеширование паролей и работа с JWT-токенами.

Хеширование паролей построено по схеме Dropbox (SHA-256 + bcrypt):
сначала пароль хешируется SHA-256, и уже 32-байтный дайджест передаётся в
bcrypt. Это снимает ограничение bcrypt в 72 байта (с версии 5.0 превышение
этого лимита приводит к ошибке) и защищает от truncation attacks.
Источник: https://dropbox.tech/security/how-dropbox-securely-stores-your-passwords

JWT-payload содержит минимум необходимых полей: `sub` (user_id), `role`, `exp`.
Email и username в токен не кладём — они могут меняться, и токен бы устаревал
некорректно.
"""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import settings

# Стоимость bcrypt: 12 раундов даёт ~250 мс на хеш — разумный баланс
# между сопротивлением brute-force и UX логина.
_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """
    Хеширует пароль по схеме Dropbox (SHA-256 + bcrypt).

    См. https://dropbox.tech/security/how-dropbox-securely-stores-your-passwords
    """
    sha256_digest = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(sha256_digest, bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Проверяет пароль против сохранённого хеша (SHA-256 + bcrypt)."""
    sha256_digest = hashlib.sha256(plain.encode("utf-8")).digest()
    return bcrypt.checkpw(sha256_digest, hashed.encode("utf-8"))


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    """Создаёт JWT с payload {sub, role, exp}. TTL берётся из settings.JWT_EXPIRE_MINUTES."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Декодирует JWT и возвращает payload. При невалидности — HTTP 401."""
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или просроченный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )
