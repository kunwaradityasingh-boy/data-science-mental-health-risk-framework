from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert (
        data["service"]
        == "mental-health-risk-research-api"
    )


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_prediction_rejects_missing_features():

    response = client.post(
        "/predict",
        json={
            "text": "Example research text.",
            "subject_id": "SUBJ-API-001",
        },
    )

    assert response.status_code == 422


def test_prediction_rejects_extra_fields():

    response = client.post(
        "/predict",
        json={
            "text": "Example research text.",
            "subject_id": "SUBJ-API-002",
            "engineered_features": {},
            "unauthorized_field": "blocked",
        },
    )

    assert response.status_code == 422