from datetime import datetime
from pydantic import BaseModel, Field


class BehavioralObservation(BaseModel):
    subject_id: str = Field(..., min_length=3)
    timestamp: datetime

    sleep_duration_hours: float | None = Field(
        default=None,
        ge=0,
        le=24
    )

    phone_usage_minutes: float | None = Field(
        default=None,
        ge=0
    )

    location_count: int | None = Field(
        default=None,
        ge=0
    )

    conversation_minutes: float | None = Field(
        default=None,
        ge=0
    )

    physical_activity_minutes: float | None = Field(
        default=None,
        ge=0
    )