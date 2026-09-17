from datetime import datetime, timezone
import math
from typing import Any

from src.data.dataset_schema import StandardizedDatasetRecord
from src.data.schemas import Modality


class DataQualityError(ValueError):
    """Raised when a data record fails quality checks."""


def validate_record(record: StandardizedDatasetRecord) -> None:
    """
    Validate a standardized record against framework-level
    data-quality rules.

    Raises:
        DataQualityError: if the record violates a quality rule.
    """

    # 1. Timestamp must not be in the future.
    now = datetime.now(timezone.utc)

    timestamp = record.timestamp

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    if timestamp > now:
        raise DataQualityError(
            "timestamp cannot be in the future"
        )

    # 2. Feature values must be finite numbers when numeric.
    for feature_name, feature_value in record.features.items():

        if isinstance(feature_value, (int, float)):
            if not math.isfinite(feature_value):
                raise DataQualityError(
                    f"feature '{feature_name}' must be finite"
                )

    # 3. Text-specific checks.
    if record.modality == Modality.TEXT:

        text = record.features.get("text")

        if text is not None and not isinstance(text, str):
            raise DataQualityError(
                "text feature must be a string"
            )

        if isinstance(text, str) and not text.strip():
            raise DataQualityError(
                "text cannot be empty"
            )

    # 4. Voice-specific checks.
    if record.modality == Modality.VOICE:

        duration = record.features.get("duration_seconds")

        if duration is not None:
            if not isinstance(duration, (int, float)):
                raise DataQualityError(
                    "voice duration must be numeric"
                )

            if duration <= 0:
                raise DataQualityError(
                    "voice duration must be greater than zero"
                )

    # 5. Behavioral-specific checks.
    if record.modality == Modality.BEHAVIOR:

        value = record.features.get("value")

        if value is not None:

            if not isinstance(value, (int, float)):
                raise DataQualityError(
                    "behavior value must be numeric"
                )

            if not math.isfinite(value):
                raise DataQualityError(
                    "behavior value must be finite"
                )


def check_duplicate_ids(
    records: list[StandardizedDatasetRecord],
) -> list[str]:
    """
    Return duplicate record IDs.

    Does not modify the input records.
    """

    seen: set[str] = set()
    duplicates: set[str] = set()

    for record in records:

        if record.record_id in seen:
            duplicates.add(record.record_id)

        seen.add(record.record_id)

    return sorted(duplicates)