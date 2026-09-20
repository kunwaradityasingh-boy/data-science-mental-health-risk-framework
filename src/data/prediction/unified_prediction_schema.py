"""
Unified prediction schema for the Early Mental Health Risk Framework.

Research-only schema.
The output is a model-generated risk signal and is NOT a clinical diagnosis.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    BEHAVIOR = "behavior"


class PredictionStatus(str, Enum):
    SUCCESS = "success"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_DATA = "insufficient_data"
    MODEL_ERROR = "model_error"


class DataQuality(str, Enum):
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class RiskSignal(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class ModalityPrediction(BaseModel):
    """Standard prediction returned by one modality."""

    model_config = ConfigDict(use_enum_values=True)

    subject_id: str = Field(..., min_length=1)

    modality: Modality

    model_id: str = Field(..., min_length=1)

    model_version: str = Field(..., min_length=1)

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Research risk signal; not a clinical probability.",
    )

    prediction: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
    )

    data_quality: DataQuality = DataQuality.UNKNOWN

    status: PredictionStatus = PredictionStatus.UNKNOWN

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    message: Optional[str] = None


class UnifiedPrediction(BaseModel):
    """Framework-level multimodal prediction."""

    model_config = ConfigDict(use_enum_values=True)

    subject_id: str = Field(..., min_length=1)

    overall_status: PredictionStatus = PredictionStatus.UNKNOWN

    risk_signal: RiskSignal = RiskSignal.UNKNOWN

    human_review_required: bool = True

    modalities_available: List[Modality] = Field(
        default_factory=list
    )

    modality_predictions: List[ModalityPrediction] = Field(
        default_factory=list
    )

    model_versions: List[str] = Field(
        default_factory=list
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    framework_version: str = "0.1.0"

    message: Optional[str] = None


def create_unknown_prediction(
    subject_id: str,
    message: str,
) -> UnifiedPrediction:
    """Return a safe UNKNOWN result when inference is unavailable."""

    return UnifiedPrediction(
        subject_id=subject_id,
        overall_status=PredictionStatus.UNKNOWN,
        risk_signal=RiskSignal.UNKNOWN,
        human_review_required=True,
        modalities_available=[],
        modality_predictions=[],
        model_versions=[],
        message=message,
    )


if __name__ == "__main__":
    prediction = UnifiedPrediction(
        subject_id="demo_subject_001",
        overall_status=PredictionStatus.SUCCESS,
        risk_signal=RiskSignal.MODERATE,
        human_review_required=True,
        modalities_available=[
            Modality.TEXT,
            Modality.VOICE,
        ],
        model_versions=[
            "dreaddit_text_baseline@1.0.0",
            "eatd_voice_logistic_regression@1.0.0",
        ],
        modality_predictions=[
            ModalityPrediction(
                subject_id="demo_subject_001",
                modality=Modality.TEXT,
                model_id="dreaddit_text_baseline",
                model_version="1.0.0",
                score=0.62,
                prediction=1,
                data_quality=DataQuality.ACCEPTABLE,
                status=PredictionStatus.SUCCESS,
            ),
            ModalityPrediction(
                subject_id="demo_subject_001",
                modality=Modality.VOICE,
                model_id="eatd_voice_logistic_regression",
                model_version="1.0.0",
                score=0.31,
                prediction=0,
                data_quality=DataQuality.ACCEPTABLE,
                status=PredictionStatus.SUCCESS,
            ),
        ],
        message="Research-only multimodal risk signal.",
    )

    print("UNIFIED PREDICTION SCHEMA: VALID")
    print(prediction.model_dump_json(indent=2))