from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_emergency_conversation() -> None:
    payload = {
        "conversation_id": "demo-001",
        "messages": [
            {
                "speaker": "patient",
                "text": "I have crushing chest pain and shortness of breath. It started suddenly.",
            },
            {
                "speaker": "provider",
                "text": "Are you dizzy or sweating? Please seek emergency care immediately.",
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json=payload)
    body = response.json()

    assert response.status_code == 201
    assert body["urgency"]["level"] == "Emergency"
    assert body["urgency"]["confidence"] >= 0.9
    assert any(entity["label"] == "SYMPTOM" for entity in body["entities"])
    assert body["summary"]


def test_validation_rejects_empty_messages() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json={"messages": []})
    assert response.status_code == 422
