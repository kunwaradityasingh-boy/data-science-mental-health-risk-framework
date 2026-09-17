from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from src.data.schemas import Modality


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class StandardizedDatasetRecord(BaseModel):
    """
    Common representation for observations coming from
    different research datasets.

    The schema intentionally keeps the target label optional because
    different datasets may use different labeling strategies.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )

    subject_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )

    timestamp: datetime = Field(
        ...,
        description="Timestamp associated with the observation.",
    )

    modality: Modality

    data_source: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Standardized features generated from the observation.",
    )

    label: str | None = Field(
        default=None,
        max_length=100,
        description="Dataset-specific target label, when available.",
    )

    label_source: str | None = Field(
        default=None,
        max_length=200,
        description="Origin or methodology used to assign the label.",
    )

    split: DatasetSplit | None = None