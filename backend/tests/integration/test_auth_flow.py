"""Интеграционные тесты flow аутентификации (api/auth.py)."""
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient


def test_register_success(client: TestClient, db_session: Any) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "Strong123!",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "newuser@example.com"
    assert body["username"] == "newuser"
    assert body["role"] == "user"
    assert "id" in body
    assert "created_at" in body
    # Хеш пароля не должен утекать в ответ.
    assert "password_hash" not in body
    assert "password" not in body


def test_register_duplicate_email(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    existing = test_user(email="taken@example.com")["user"]
    response = client.post(
        "/api/auth/register",
        json={
            "email": existing.email,
            "username": "different",
            "password": "Strong123!",
        },
    )
    assert response.status_code == 400
    assert "email" in response.json()["detail"].lower()


def test_register_duplicate_username(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    existing = test_user(username="taken")["user"]
    response = client.post(
        "/api/auth/register",
        json={
            "email": "different@example.com",
            "username": existing.username,
            "password": "Strong123!",
        },
    )
    assert response.status_code == 400
    assert "username" in response.json()["detail"].lower()


def test_login_success(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    created = test_user(username="loginuser", password="Strong123!")
    response = client.post(
        "/api/auth/login",
        json={"username_or_email": "loginuser", "password": "Strong123!"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and len(body["access_token"]) > 0
    assert body["user"]["username"] == "loginuser"
    assert body["user"]["id"] == str(created["user"].id)


def test_login_wrong_password(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    test_user(username="alice", password="Correct123!")
    response = client.post(
        "/api/auth/login",
        json={"username_or_email": "alice", "password": "Wrong456!"},
    )
    assert response.status_code == 401


def test_login_unknown_user_same_message_as_wrong_password(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
) -> None:
    """Защита от перебора: ответ для несуществующего пользователя и для неверного
    пароля должен быть идентичным (status + body)."""
    test_user(username="known", password="Correct123!")
    wrong_password = client.post(
        "/api/auth/login",
        json={"username_or_email": "known", "password": "Wrong456!"},
    )
    unknown_user = client.post(
        "/api/auth/login",
        json={"username_or_email": "ghost", "password": "Whatever1!"},
    )
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_get_me_with_token(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user(email="me@example.com", username="meuser")["user"]
    response = client.get("/api/auth/me", headers=auth_headers(user))
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["email"] == "me@example.com"
    assert body["username"] == "meuser"
    assert "password_hash" not in body


def test_get_me_without_token(client: TestClient) -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_update_profile(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user(email="old@example.com", username="oldname")["user"]
    response = client.put(
        "/api/auth/me",
        headers=auth_headers(user),
        json={"email": "new@example.com", "username": "newname"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "new@example.com"
    assert body["username"] == "newname"
    assert body["id"] == str(user.id)


def test_change_password(
    client: TestClient,
    test_user: Callable[..., dict[str, Any]],
    auth_headers: Callable[[Any], dict[str, str]],
) -> None:
    user = test_user(username="pwduser", password="OldPass123!")["user"]

    change = client.put(
        "/api/auth/password",
        headers=auth_headers(user),
        json={"current_password": "OldPass123!", "new_password": "NewPass456!"},
    )
    assert change.status_code == 204

    # Новый пароль работает.
    login_new = client.post(
        "/api/auth/login",
        json={"username_or_email": "pwduser", "password": "NewPass456!"},
    )
    assert login_new.status_code == 200

    # Старый пароль больше не работает.
    login_old = client.post(
        "/api/auth/login",
        json={"username_or_email": "pwduser", "password": "OldPass123!"},
    )
    assert login_old.status_code == 401
