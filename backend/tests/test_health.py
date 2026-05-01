"""Тесты health-эндпоинта."""
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    """GET /api/health должен возвращать 200 и фиксированный JSON."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backend"}
