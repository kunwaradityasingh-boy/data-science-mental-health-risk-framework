from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    BEHAVIOR = "behavior"


class CommonMetadata(BaseModel):
    """
    Metadata shared by all incoming data records.
    """

    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Pseudonymous identifier; must not contain direct PII.",
    )

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp associated with the observation.",
    )

    consent_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Reference to the applicable consent record.",
    )

    modality: Modality


class TextRecord(CommonMetadata):
    """
    Schema for text observations.
    """

    modality: Modality = Modality.TEXT

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    language: str = Field(
        default="en",
        min_length=2,
        max_length=10,
    )

    source_type: str = Field(
        default="unknown",
        max_length=50,
    )


class VoiceRecord(CommonMetadata):
    """
    Schema for voice/audio observations.
    """

    modality: Modality = Modality.VOICE

    audio_uri: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    duration_seconds: float = Field(
        ...,
        gt=0,
        le=3600,
    )

    sample_rate_hz: int = Field(
        ...,
        gt=0,
        le=192000,
    )

    channels: int = Field(
        default=1,
        ge=1,
        le=8,
    )

    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class BehaviorRecord(CommonMetadata):
    """
    Schema for longitudinal behavioral observations.
    """

    modality: Modality = Modality.BEHAVIOR

    event_type: str = Field(
        ...,
        min_length=2,
        max_length=100,
    )

    value: float = Field(
        ...,
        description="Numeric value associated with the behavioral event.",
    )

    unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    source_system: str = Field(
        default="unknown",
        max_length=100,
    )