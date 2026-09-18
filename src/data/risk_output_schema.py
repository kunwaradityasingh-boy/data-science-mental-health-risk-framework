from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class ReviewRecommendation(str, Enum):
    NOT_REQUIRED = "not_required"
    RECOMMENDED = "recommended"
    REQUIRED = "required"
    UNAVAILABLE = "unavailable"


class DataQualityStatus(str, Enum):
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class RiskOutput(BaseModel):
    """
    Structured research risk output.

    This schema represents a model-generated research signal.
    It is not a clinical diagnosis.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )

    subject_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Pseudonymous subject identifier.",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    risk_level: RiskLevel = Field(
        ...,
        description=(
            "Research risk signal; not a clinical diagnosis."
        ),
    )

    model_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Model-generated score. "
            "Must not automatically be interpreted "
            "as clinical probability."
        ),
    )

    model_version: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    available_modalities: list[str] = Field(
        default_factory=list,
    )

    missing_modalities: list[str] = Field(
        default_factory=list,
    )

    data_quality: DataQualityStatus = (
        DataQualityStatus.UNKNOWN
    )

    contributing_evidence: list[str] = Field(
        default_factory=list,
    )

    review_recommendation: ReviewRecommendation = (
        ReviewRecommendation.UNAVAILABLE
    )

    is_clinical_diagnosis: bool = False

    disclaimer: str = Field(
        default=(
            "Research risk signal only. "
            "This output is not a clinical diagnosis "
            "and should not be used as the sole basis "
            "for medical, academic, employment, insurance, "
            "legal, or emergency decisions."
        ),
        min_length=20,
        max_length=1000,
    )