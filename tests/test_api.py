from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint_reports_service_status() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "vehicle-insurance-api"
    assert isinstance(body["ml_available"], bool)


def test_training_status_endpoint_is_available() -> None:
    response = client.get("/api/train/status")
    assert response.status_code == 200
    assert "status" in response.json()
