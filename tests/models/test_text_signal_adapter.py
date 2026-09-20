from unittest.mock import Mock

import pytest

from src.data.modality_signal_schema import (
    Modality,
    SignalStatus,
)
from src.models.text_signal_adapter import (
    TextSignalAdapter,
)


def test_text_adapter_returns_insufficient_without_engineered_features():
    service = Mock()
    service.model_version = "text-engineered-logistic-v1"

    adapter = TextSignalAdapter(
        prediction_service=service
    )

    signal = adapter.predict(
        text="I feel stressed and unable to concentrate.",
        subject_id="TEST-001",
        engineered_features=None,
    )

    assert signal.modality == Modality.TEXT
    assert signal.score is None
    assert signal.status == SignalStatus.INSUFFICIENT
    assert signal.data_quality == "insufficient"

    service.predict_text.assert_not_called()


def test_text_adapter_converts_prediction_to_modality_signal():
    service = Mock()
    service.model_version = "text-engineered-logistic-v1"

    fake_result = Mock()
    fake_result.model_version = "text-engineered-logistic-v1"
    fake_result.model_score = 0.73
    fake_result.data_quality.value = "acceptable"
    fake_result.contributing_evidence = [
        "text-derived TF-IDF features",
        "engineered linguistic features",
    ]

    service.predict_text.return_value = fake_result

    adapter = TextSignalAdapter(
        prediction_service=service
    )

    engineered_features = object()

    signal = adapter.predict(
        text="I feel stressed and unable to concentrate.",
        subject_id="TEST-001",
        engineered_features=engineered_features,
    )

    assert signal.modality == Modality.TEXT
    assert signal.model_version == "text-engineered-logistic-v1"
    assert signal.score == 0.73
    assert signal.status == SignalStatus.AVAILABLE
    assert signal.data_quality == "acceptable"

    assert (
        "text-derived TF-IDF features"
        in signal.evidence
    )

    service.predict_text.assert_called_once()


def test_text_adapter_rejects_empty_text():
    service = Mock()
    service.model_version = "text-engineered-logistic-v1"

    adapter = TextSignalAdapter(
        prediction_service=service
    )

    with pytest.raises(Exception):
        adapter.predict(
            text="",
            subject_id="TEST-001",
            engineered_features=None,
        )


def test_text_adapter_rejects_empty_subject_id():
    service = Mock()
    service.model_version = "text-engineered-logistic-v1"

    adapter = TextSignalAdapter(
        prediction_service=service
    )

    with pytest.raises(Exception):
        adapter.predict(
            text="I feel stressed.",
            subject_id="",
            engineered_features=None,
        )