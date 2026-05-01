"""
API-эндпоинты аутентификации.

См. .knowledge/architecture/api-contract.md, раздел 1.

Особенности безопасности:
- При логине не различаем "пользователь не найден" и "неверный пароль" — единый
  ответ 401 с общим сообщением, чтобы не давать оракул для перебора.
- При смене пароля ранее выданные JWT остаются валидными до своего exp:
  revocation list осознанно не делаем (см. .knowledge/project/scope.md).
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.repositories import user_repo
from app.schemas.auth import LoginRequest, TokenResponse
from app.schemas.user import (
    PasswordChange,
    UserCreate,
    UserResponse,
    UserUpdate,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    if user_repo.get_user_by_email(db, payload.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким email уже существует",
        )
    if user_repo.get_user_by_username(db, payload.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким username уже существует",
        )
    return user_repo.create_user(
        db,
        email=payload.email,
        username=payload.username,
        password_hash=hash_password(payload.password),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    # Принимаем и email, и username — определяем по наличию '@'.
    if "@" in payload.username_or_email:
        user = user_repo.get_user_by_email(db, payload.username_or_email)
    else:
        user = user_repo.get_user_by_username(db, payload.username_or_email)

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
        )

    token = create_access_token(user.id, user.role)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout() -> Response:
    # Stateless JWT: фронт сам удаляет токен из localStorage.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.put("/me", response_model=UserResponse)
def update_me(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    fields: dict[str, str] = {}
    if payload.email is not None and payload.email != current_user.email:
        if user_repo.get_user_by_email(db, payload.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email уже занят",
            )
        fields["email"] = payload.email
    if payload.username is not None and payload.username != current_user.username:
        if user_repo.get_user_by_username(db, payload.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username уже занят",
            )
        fields["username"] = payload.username

    if fields:
        fields["updated_at"] = datetime.now(timezone.utc)
        user_repo.update_user(db, current_user, fields)
    return current_user


@router.put("/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Текущий пароль неверен",
        )
    user_repo.update_user(
        db,
        current_user,
        {
            "password_hash": hash_password(payload.new_password),
            "updated_at": datetime.now(timezone.utc),
        },
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
