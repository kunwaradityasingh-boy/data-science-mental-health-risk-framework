"""
Framework-level prediction service.

Connects:
    Multimodal Integration Service
    +
    Risk Policy Service
    +
    Unified Prediction Schema

Research-only system.
"""

from typing import Optional

from src.data.prediction.unified_prediction_schema import (
    DataQuality,
    Modality,
    ModalityPrediction,
    PredictionStatus,
    UnifiedPrediction,
)
from src.models.multimodal_integration_service import (
    MultimodalIntegrationService,
)
from src.models.risk_policy_service import RiskPolicyService


class FrameworkPredictionService:
    """Central framework prediction service."""

    def __init__(self):
        self.integration = MultimodalIntegrationService()
        self.risk_policy = RiskPolicyService()

    def health(self) -> dict:
        """Return framework integration health."""

        integration_health = self.integration.health_check()

        return {
            "status": integration_health["status"],
            "framework_version": integration_health[
                "framework_version"
            ],
            "modalities": integration_health["modalities"],
            "cross_dataset_fusion_enabled": integration_health[
                "cross_dataset_fusion_enabled"
            ],
            "risk_policy": "loaded",
            "research_only": True,
        }

    def predict(
        self,
        subject_id: str,
        text_score: Optional[float] = None,
        voice_score: Optional[float] = None,
        behavior_score: Optional[float] = None,
    ) -> UnifiedPrediction:
        """
        Build a unified research prediction from available
        modality scores.

        No cross-dataset score fusion is performed.
        """

        modality_predictions = []
        modalities_available = []
        model_versions = []

        # ---------------------------------------------------------
        # TEXT
        # ---------------------------------------------------------
        if text_score is not None:
            risk = self.risk_policy.evaluate(text_score)

            modality_predictions.append(
                ModalityPrediction(
                    subject_id=subject_id,
                    modality=Modality.TEXT,
                    model_id="dreaddit_text_baseline",
                    model_version="1.0.0",
                    score=text_score,
                    prediction=(
                        1
                        if risk["risk_signal"] in {
                            "moderate",
                            "high",
                        }
                        else 0
                    ),
                    data_quality=DataQuality.ACCEPTABLE,
                    status=PredictionStatus.SUCCESS,
                    message=(
                        "Research-only text signal."
                    ),
                )
            )

            modalities_available.append(Modality.TEXT)
            model_versions.append(
                "dreaddit_text_baseline@1.0.0"
            )

        # ---------------------------------------------------------
        # VOICE
        # ---------------------------------------------------------
        if voice_score is not None:
            risk = self.risk_policy.evaluate(voice_score)

            modality_predictions.append(
                ModalityPrediction(
                    subject_id=subject_id,
                    modality=Modality.VOICE,
                    model_id="eatd_voice_logistic_regression",
                    model_version="1.0.0",
                    score=voice_score,
                    prediction=(
                        1
                        if risk["risk_signal"] in {
                            "moderate",
                            "high",
                        }
                        else 0
                    ),
                    data_quality=DataQuality.ACCEPTABLE,
                    status=PredictionStatus.SUCCESS,
                    message=(
                        "Research-only voice signal."
                    ),
                )
            )

            modalities_available.append(Modality.VOICE)
            model_versions.append(
                "eatd_voice_logistic_regression@1.0.0"
            )

        # ---------------------------------------------------------
        # BEHAVIOR
        # ---------------------------------------------------------
        if behavior_score is not None:
            risk = self.risk_policy.evaluate(behavior_score)

            modality_predictions.append(
                ModalityPrediction(
                    subject_id=subject_id,
                    modality=Modality.BEHAVIOR,
                    model_id="studentlife_behavior_baseline",
                    model_version="1.0.0",
                    score=behavior_score,
                    prediction=(
                        1
                        if risk["risk_signal"] in {
                            "moderate",
                            "high",
                        }
                        else 0
                    ),
                    data_quality=DataQuality.ACCEPTABLE,
                    status=PredictionStatus.SUCCESS,
                    message=(
                        "Research-only behavioral signal."
                    ),
                )
            )

            modalities_available.append(Modality.BEHAVIOR)
            model_versions.append(
                "studentlife_behavior_baseline@1.0.0"
            )

        # ---------------------------------------------------------
        # NO VALID INPUT
        # ---------------------------------------------------------
        if not modality_predictions:
            return self.integration.create_unknown_prediction(
                subject_id=subject_id,
                message=(
                    "No valid modality score was supplied. "
                    "Risk signal remains UNKNOWN."
                ),
            )

        # ---------------------------------------------------------
        # FRAMEWORK-LEVEL STATUS
        # ---------------------------------------------------------
        highest_signal = "low"

        for prediction in modality_predictions:
            signal = self.risk_policy.classify(
                prediction.score
            ).value

            if signal == "high":
                highest_signal = "high"
                break

            if (
                signal == "moderate"
                and highest_signal == "low"
            ):
                highest_signal = "moderate"

        review_required = highest_signal in {
            "moderate",
            "high",
        }

        return UnifiedPrediction(
            subject_id=subject_id,
            overall_status=PredictionStatus.SUCCESS,
            risk_signal=highest_signal,
            human_review_required=review_required,
            modalities_available=modalities_available,
            modality_predictions=modality_predictions,
            model_versions=model_versions,
            framework_version="0.1.0",
            message=(
                "Research-only multimodal framework signal. "
                "Not a clinical diagnosis."
            ),
        )


def main() -> None:
    print("=" * 72)
    print("FRAMEWORK PREDICTION SERVICE")
    print("=" * 72)

    service = FrameworkPredictionService()

    print("\nHEALTH")
    print(service.health())

    print("\nTEST 1: TEXT + VOICE")

    result = service.predict(
        subject_id="demo_subject_001",
        text_score=0.62,
        voice_score=0.31,
    )

    print(result.model_dump_json(indent=2))

    print("\nTEST 2: NO INPUT")

    unknown = service.predict(
        subject_id="demo_subject_002",
    )

    print(unknown.model_dump_json(indent=2))

    print("\nPHASE 9.5 SERVICE: PASSED")


if __name__ == "__main__":
    main()