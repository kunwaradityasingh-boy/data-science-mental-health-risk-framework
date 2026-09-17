from datetime import datetime, timedelta, timezone

import pytest

from src.data.data_quality import (
    DataQualityError,
    check_duplicate_ids,
    validate_record,
)
from src.data.dataset_schema import StandardizedDatasetRecord


def make_record(record_id: str, value: float):
    timestamp = datetime.now(timezone.utc) - timedelta(hours=1)

    return StandardizedDatasetRecord(
        record_id=record_id,
        subject_id="user_001",
        timestamp=timestamp,
        modality="behavior",
        data_source="research_dataset",
        features={"value": value},
    )


def test_valid_record_passes():
    record = make_record("behavior_001", 6.2)

    validate_record(record)


def test_nan_value_is_rejected():
    record = make_record("behavior_002", float("nan"))

    with pytest.raises(DataQualityError):
        validate_record(record)


def test_duplicate_ids_are_detected():
    records = [
        make_record("behavior_001", 6.2),
        make_record("behavior_001", 7.1),
        make_record("behavior_002", 5.5),
    ]

    duplicates = check_duplicate_ids(records)

    assert duplicates == ["behavior_001"]