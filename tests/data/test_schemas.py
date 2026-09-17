from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data.schemas import (
    BehaviorRecord,
    TextRecord,
    VoiceRecord,
)
from src.data.dataset_schema import StandardizedDatasetRecord
from src.data.label_schema import LabelDefinition
from src.data.consent_schema import ConsentRecord


def test_text_record_valid():
    record = TextRecord(
        subject_id="user_001",
        timestamp=datetime.now(timezone.utc),
        consent_id="consent_001",
        text="This is a sample text.",
    )

    assert record.text == "This is a sample text."


def test_voice_record_valid():
    record = VoiceRecord(
        subject_id="user_001",
        timestamp=datetime.now(timezone.utc),
        consent_id="consent_001",
        audio_uri="audio/sample.wav",
        duration_seconds=30.5,
        sample_rate_hz=16000,
    )

    assert record.duration_seconds == 30.5


def test_behavior_record_valid():
    record = BehaviorRecord(
        subject_id="user_001",
        timestamp=datetime.now(timezone.utc),
        consent_id="consent_001",
        event_type="screen_time",
        value=6.2,
        unit="hours",
    )

    assert record.value == 6.2


def test_dataset_record_valid():
    record = StandardizedDatasetRecord(
        record_id="record_001",
        subject_id="user_001",
        timestamp=datetime.now(timezone.utc),
        modality="behavior",
        data_source="research_dataset",
        features={"screen_time": 6.2},
        label="high-risk",
    )

    assert record.record_id == "record_001"


def test_label_definition_valid():
    label = LabelDefinition(
        original_label="high-risk",
        normalized_risk="high",
        task_type="binary_risk",
        label_source="research_dataset",
    )

    assert label.normalized_risk.value == "high"


def test_consent_record_valid():
    consent = ConsentRecord(
        consent_id="consent_001",
        subject_id="user_001",
        purpose="research",
        modalities_allowed=["text", "behavior"],
        granted_at=datetime.now(timezone.utc),
    )

    assert consent.status.value == "active"


def test_invalid_text_record_is_rejected():
    with pytest.raises(ValidationError):
        TextRecord(
            subject_id="u",
            timestamp=datetime.now(timezone.utc),
            consent_id="c",
            text="hello",
        )