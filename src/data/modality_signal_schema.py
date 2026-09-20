from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    TEXT = "text"
    BEHAVIOR = "behavior"
    VOICE = "voice"


class SignalStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT = "insufficient"


class ModalitySignal(BaseModel):
    """
    Standardized research signal produced by one modality.

    This is a modality-level research representation.
    It is not a clinical diagnosis or clinical probability.
    """

    model_config = ConfigDict(extra="forbid")

    modality: Modality = Field(
        ...,
        description="Source modality producing the signal.",
    )

    model_version: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized model-generated research score. "
            "Not automatically a clinical probability."
        ),
    )

    status: SignalStatus = Field(
        default=SignalStatus.AVAILABLE,
    )

    data_quality: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    evidence: list[str] = Field(
        default_factory=list,
    )

    limitations: list[str] = Field(
        default_factory=list,
    )