from __future__ import annotations

from src.data.modality_signal_schema import (
    Modality,
    ModalitySignal,
    SignalStatus,
)


class BehavioralSignalAdapter:
    """
    Converts the current StudentLife behavioral baseline
    evaluation into the common ModalitySignal format.

    The current behavioral baseline does not establish a
    reliable normalized predictive risk score.

    Research-only. Not a clinical diagnosis.
    """

    def __init__(
        self,
        model_version: str = "studentlife-behavioral-baseline-v1",
    ) -> None:
        self.model_version = model_version

    def get_signal(self) -> ModalitySignal:
        """
        Return the current validated behavioral signal state.
        """

        return ModalitySignal(
            modality=Modality.BEHAVIOR,
            model_version=self.model_version,
            score=None,
            status=SignalStatus.INSUFFICIENT,
            data_quality="insufficient",
            evidence=[
                (
                    "StudentLife behavioral features were "
                    "evaluated at the participant level."
                ),
                (
                    "Group-aware validation was used to prevent "
                    "participant leakage."
                ),
                (
                    "The current behavioral models did not "
                    "outperform the participant-level mean baseline."
                ),
            ],
            limitations=[
                (
                    "No reliable normalized behavioral risk score "
                    "is established by the current baseline."
                ),
                (
                    "The current StudentLife formulation is not "
                    "sufficient for a validated predictive risk signal."
                ),
                (
                    "This is a research-only representation and "
                    "not a clinical diagnosis."
                ),
            ],
        )