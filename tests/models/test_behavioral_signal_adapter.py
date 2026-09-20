from src.data.modality_signal_schema import (
    Modality,
    SignalStatus,
)
from src.models.behavioral_signal_adapter import (
    BehavioralSignalAdapter,
)


def test_behavioral_adapter_returns_insufficient_signal():
    adapter = BehavioralSignalAdapter()

    signal = adapter.get_signal()

    assert signal.modality == Modality.BEHAVIOR
    assert signal.model_version == (
        "studentlife-behavioral-baseline-v1"
    )
    assert signal.score is None
    assert signal.status == SignalStatus.INSUFFICIENT
    assert signal.data_quality == "insufficient"


def test_behavioral_adapter_contains_validation_evidence():
    adapter = BehavioralSignalAdapter()

    signal = adapter.get_signal()

    assert any(
        "participant level" in evidence.lower()
        for evidence in signal.evidence
    )

    assert any(
        "leakage" in evidence.lower()
        for evidence in signal.evidence
    )


def test_behavioral_adapter_documents_baseline_limitation():
    adapter = BehavioralSignalAdapter()

    signal = adapter.get_signal()

    assert any(
        "did not outperform" in evidence.lower()
        for evidence in signal.evidence
    )

    assert any(
        "reliable normalized" in limitation.lower()
        for limitation in signal.limitations
    )