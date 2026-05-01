"""Общие фикстуры pytest."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """HTTP-клиент к FastAPI-приложению (in-process, без поднятия сервера)."""
    return TestClient(app)
