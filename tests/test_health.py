from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_status_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_body():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_check_content_type():
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
