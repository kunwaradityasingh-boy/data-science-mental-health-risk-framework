from datetime import datetime

from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class ConsentStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ConsentPurpose(str, Enum):
    RESEARCH = "research"
    SCREENING = "screening"
    MODEL_EVALUATION = "model_evaluation"


class ConsentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consent_id: str = Field(
        ...,
        min_length=3,
        max_length=100
    )

    subject_id: str = Field(
        ...,
        min_length=3,
        max_length=100
    )

    purpose: ConsentPurpose

    modalities_allowed: list[str] = Field(
        ...,
        min_length=1
    )

    granted_at: datetime

    expires_at: datetime | None = None

    revoked_at: datetime | None = None

    status: ConsentStatus = ConsentStatus.ACTIVE