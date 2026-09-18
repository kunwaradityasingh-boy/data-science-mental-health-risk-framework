from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskLevel,
    RiskOutput,
    ReviewRecommendation,
)


def test_valid_risk_output():

    output = RiskOutput(
        prediction_id="PRED-000001",
        subject_id="SUBJ-001",
        timestamp=datetime.now(timezone.utc),
        risk_level=RiskLevel.MODERATE,
        model_score=0.72,
        model_version="text-engineered-logistic-v1",
        available_modalities=[
            "text",
            "behavior",
        ],
        missing_modalities=[
            "voice",
        ],
        data_quality=DataQualityStatus.ACCEPTABLE,
        contributing_evidence=[
            "text-derived features",
            "linguistic feature groups",
        ],
        review_recommendation=(
            ReviewRecommendation.RECOMMENDED
        ),
    )

    assert output.prediction_id == "PRED-000001"
    assert output.subject_id == "SUBJ-001"
    assert output.risk_level == RiskLevel.MODERATE
    assert output.model_score == 0.72
    assert output.is_clinical_diagnosis is False


def test_model_score_must_be_between_zero_and_one():

    with pytest.raises(ValidationError):

        RiskOutput(
            prediction_id="PRED-000002",
            subject_id="SUBJ-002",
            risk_level=RiskLevel.HIGH,
            model_score=1.5,
            model_version="test-v1",
        )


def test_negative_model_score_rejected():

    with pytest.raises(ValidationError):

        RiskOutput(
            prediction_id="PRED-000003",
            subject_id="SUBJ-003",
            risk_level=RiskLevel.LOW,
            model_score=-0.1,
            model_version="test-v1",
        )


def test_unknown_risk_can_have_no_score():

    output = RiskOutput(
        prediction_id="PRED-000004",
        subject_id="SUBJ-004",
        risk_level=RiskLevel.UNKNOWN,
        model_score=None,
        model_version="test-v1",
        data_quality=(
            DataQualityStatus.INSUFFICIENT
        ),
        review_recommendation=(
            ReviewRecommendation.UNAVAILABLE
        ),
    )

    assert output.risk_level == RiskLevel.UNKNOWN
    assert output.model_score is None


def test_extra_fields_are_rejected():

    with pytest.raises(ValidationError):

        RiskOutput(
            prediction_id="PRED-000005",
            subject_id="SUBJ-005",
            risk_level=RiskLevel.LOW,
            model_score=0.2,
            model_version="test-v1",
            unauthorized_field="should_fail",
        )