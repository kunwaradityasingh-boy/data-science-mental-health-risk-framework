from pydantic import ValidationError
import pytest

from src.data.modality_signal_schema import (
    Modality,
    ModalitySignal,
    SignalStatus,
)


def test_text_signal():

    signal = ModalitySignal(
        modality=Modality.TEXT,
        model_version="dreaddit-text-v1",
        score=0.72,
        status=SignalStatus.AVAILABLE,
        data_quality="acceptable",
        evidence=["text_model_signal"],
        limitations=[],
    )

    assert signal.modality == Modality.TEXT
    assert signal.score == 0.72


def test_behavior_signal():

    signal = ModalitySignal(
        modality=Modality.BEHAVIOR,
        model_version="studentlife-behavior-v1",
        score=None,
        status=SignalStatus.INSUFFICIENT,
        data_quality="degraded",
        evidence=[],
        limitations=[
            "Current behavioral baseline did not outperform "
            "the participant-level mean baseline."
        ],
    )

    assert signal.modality == Modality.BEHAVIOR
    assert signal.score is None
    assert signal.status == SignalStatus.INSUFFICIENT


def test_voice_unavailable():

    signal = ModalitySignal(
        modality=Modality.VOICE,
        model_version="modma-voice-pending",
        score=None,
        status=SignalStatus.UNAVAILABLE,
        data_quality="unknown",
        evidence=[],
        limitations=["MODMA audio download is currently unavailable."],
    )

    assert signal.status == SignalStatus.UNAVAILABLE


def test_score_must_be_between_zero_and_one():

    with pytest.raises(ValidationError):

        ModalitySignal(
            modality=Modality.TEXT,
            model_version="test-v1",
            score=1.5,
            data_quality="acceptable",
        )


def test_unknown_extra_field_rejected():

    with pytest.raises(ValidationError):

        ModalitySignal(
            modality=Modality.TEXT,
            model_version="test-v1",
            score=0.5,
            data_quality="acceptable",
            unexpected_field="should_fail",
        )