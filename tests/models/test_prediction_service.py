import pandas as pd
import pytest

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskLevel,
)
from src.models.prediction_service import (
    PredictionService,
    PredictionServiceError,
)


def build_engineered_features():
    """
    Load one real Dreaddit training row and return
    its engineered feature structure.
    """

    train_path = (
        "data/raw/dreaddit/extracted/"
        "dreaddit-train.csv"
    )

    df = pd.read_csv(
        train_path
    )

    return df.iloc[[0]].copy()


def test_prediction_service_initializes():

    service = PredictionService()

    assert service.model is not None
    assert service.vectorizer is not None
    assert service.scaler is not None
    assert service.policy is not None


def test_prediction_returns_structured_output():

    service = PredictionService()

    df = build_engineered_features()

    result = service.predict_text(
        text=df.iloc[0]["text"],
        subject_id="SUBJ-TEST-001",
        engineered_features=df,
        available_modalities=[
            "text"
        ],
        missing_modalities=[
            "voice",
            "behavior",
        ],
        data_quality=(
            DataQualityStatus.ACCEPTABLE
        ),
    )

    assert result.subject_id == (
        "SUBJ-TEST-001"
    )

    assert result.model_score is not None

    assert 0.0 <= result.model_score <= 1.0

    assert result.is_clinical_diagnosis is False

    assert result.risk_level in {
        RiskLevel.LOW,
        RiskLevel.MODERATE,
        RiskLevel.HIGH,
        RiskLevel.UNKNOWN,
    }


def test_empty_text_is_rejected():

    service = PredictionService()

    df = build_engineered_features()

    with pytest.raises(
        PredictionServiceError
    ):

        service.predict_text(
            text="",
            subject_id="SUBJ-TEST-002",
            engineered_features=df,
        )


def test_missing_engineered_features_rejected():

    service = PredictionService()

    with pytest.raises(
        PredictionServiceError
    ):

        service.predict_text(
            text="Example research text.",
            subject_id="SUBJ-TEST-003",
            engineered_features=None,
        )