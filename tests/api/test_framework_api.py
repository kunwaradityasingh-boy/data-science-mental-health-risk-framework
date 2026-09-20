from fastapi.testclient import TestClient

from apps.api.framework_api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["version"] == "0.1.0"
    assert data["research_only"] is True
    assert data["clinical_diagnosis"] is False


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["framework_version"] == "0.1.0"
    assert data["risk_policy"] == "loaded"
    assert data["research_only"] is True

    assert "text" in data["modalities"]
    assert "voice" in data["modalities"]
    assert "behavior" in data["modalities"]


def test_framework_prediction_text_voice():
    response = client.post(
        "/framework/predict",
        json={
            "subject_id": "test_subject_001",
            "text_score": 0.62,
            "voice_score": 0.31,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["subject_id"] == "test_subject_001"
    assert data["overall_status"] == "success"
    assert data["risk_signal"] == "moderate"
    assert data["human_review_required"] is True

    assert data["modalities_available"] == [
        "text",
        "voice",
    ]

    assert len(data["modality_predictions"]) == 2

    assert (
        data["modality_predictions"][0]["modality"]
        == "text"
    )

    assert (
        data["modality_predictions"][1]["modality"]
        == "voice"
    )


def test_framework_prediction_unknown():
    response = client.post(
        "/framework/predict",
        json={
            "subject_id": "test_subject_002",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["overall_status"] == "unknown"
    assert data["risk_signal"] == "unknown"
    assert data["human_review_required"] is True
    assert data["modalities_available"] == []


def test_invalid_score():
    response = client.post(
        "/framework/predict",
        json={
            "subject_id": "test_subject_003",
            "text_score": 1.5,
        },
    )

    assert response.status_code == 422


def test_missing_subject_id():
    response = client.post(
        "/framework/predict",
        json={
            "text_score": 0.50,
        },
    )

    assert response.status_code == 422


if __name__ == "__main__":
    print("Framework API tests are ready.")