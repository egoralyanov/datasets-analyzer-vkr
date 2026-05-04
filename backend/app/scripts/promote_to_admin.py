"""
CLI-скрипт повышения пользователя до роли администратора.

Запуск (из контейнера backend):

    docker compose exec backend python -m app.scripts.promote_to_admin <email>

Скрипт идемпотентен: повторный запуск для уже-админа завершается успешно с
сообщением «already admin» и exit code 0. Если пользователь не найден —
exit code 1, БД не модифицируется.

Используется однократно при первичной настройке системы (первый админ),
а также для повышения дополнительных пользователей в будущем. UI для
управления ролями в Спринте 4 не делаем (см. plans/04-reports-and-history.md).
"""
from __future__ import annotations

import argparse
import sys

from app.core.db import SessionLocal
from app.repositories.user_repo import get_user_by_email


def promote(email: str) -> int:
    """
    Повышает пользователя с указанным email до role='admin'.

    Returns:
        0 — успех (повысили или уже админ).
        1 — пользователь не найден.
    """
    db = SessionLocal()
    try:
        user = get_user_by_email(db, email)
        if user is None:
            print(f"User not found: email={email}", file=sys.stderr)
            return 1
        if user.role == "admin":
            print(f"User {email} is already admin — no changes")
            return 0
        user.role = "admin"
        db.commit()
        print(f"OK: user {email} promoted to admin")
        # JWT в активной сессии содержит старую роль; authStore на фронте
        # держит снимок user из login-response. Бэкенд читает свежую роль
        # из БД на каждом запросе, но UI-навигация (пункт «Админ» в шапке)
        # обновится только после перелогина.
        print("Готово. Для применения изменений в UI выполните logout/login.")
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Повышение пользователя до роли administrator"
    )
    parser.add_argument("email", help="email пользователя для повышения")
    args = parser.parse_args()
    return promote(args.email)


if __name__ == "__main__":
    sys.exit(main())
