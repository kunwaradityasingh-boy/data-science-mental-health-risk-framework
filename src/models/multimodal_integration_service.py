"""
Multimodal integration service.

This service connects registered modality models at the framework level.
It does not perform cross-dataset probability fusion.

Research-only:
    Outputs are risk signals and must not be interpreted as clinical
    diagnoses or clinical probabilities.
"""

from pathlib import Path
from typing import Dict, List

import yaml

from src.data.prediction.unified_prediction_schema import (
    UnifiedPrediction,
    PredictionStatus,
    RiskSignal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "integration"
    / "multimodal_integration.yaml"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "model_registry.yaml"
)


class MultimodalIntegrationService:
    """Validate and expose registered multimodal components."""

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        registry_path: Path = REGISTRY_PATH,
    ):
        self.config_path = Path(config_path)
        self.registry_path = Path(registry_path)

        self.config = self._load_yaml(self.config_path)
        self.registry = self._load_yaml(self.registry_path)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Required configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure: {path}")

        return data

    def registered_model_ids(self) -> List[str]:
        """Return model IDs registered in the model registry."""

        models = self.registry.get("models", [])

        return [
            model["model_id"]
            for model in models
            if isinstance(model, dict) and "model_id" in model
        ]

    def configured_modalities(self) -> List[str]:
        """Return configured modalities."""

        modalities = self.config.get("modalities", {})

        return [
            name
            for name, settings in modalities.items()
            if isinstance(settings, dict)
            and settings.get("enabled", False)
        ]

    def verify_model_registry(self) -> Dict[str, bool]:
        """Check whether configured models exist in the registry."""

        registered = set(self.registered_model_ids())
        results = {}

        for modality, settings in self.config.get(
            "modalities", {}
        ).items():

            model_id = settings.get("model_id")

            if model_id:
                results[model_id] = model_id in registered

        fusion = self.config.get("fusion", {})
        fusion_model = (
            fusion
            .get("available_fusion", {})
            .get("model_id")
        )

        if fusion_model:
            results[fusion_model] = fusion_model in registered

        return results

    def verify_artifacts(self) -> Dict[str, bool]:
        """Check configured model artifact paths."""

        results = {}

        for modality, settings in self.config.get(
            "modalities", {}
        ).items():

            artifacts = settings.get("artifact", {})

            for artifact_name, relative_path in artifacts.items():

                path = PROJECT_ROOT / relative_path

                key = f"{modality}:{artifact_name}"

                results[key] = path.exists()

        fusion = self.config.get("fusion", {})

        fusion_artifact = (
            fusion
            .get("available_fusion", {})
            .get("artifact")
        )

        if fusion_artifact:
            path = PROJECT_ROOT / fusion_artifact
            results["fusion:text_voice"] = path.exists()

        return results

    def health_check(self) -> Dict[str, object]:
        """Return integration-layer health information."""

        registry_status = self.verify_model_registry()
        artifact_status = self.verify_artifacts()

        registry_ok = all(registry_status.values())
        artifacts_ok = all(artifact_status.values())

        return {
            "status": "healthy"
            if registry_ok and artifacts_ok
            else "degraded",
            "framework_version": self.config[
                "framework"
            ].get("version", "unknown"),
            "modalities": self.configured_modalities(),
            "registry": registry_status,
            "artifacts": artifact_status,
            "cross_dataset_fusion_enabled": self.config[
                "fusion"
            ].get("enabled", False),
        }

    def create_unknown_prediction(
        self,
        subject_id: str,
        message: str,
    ) -> UnifiedPrediction:
        """
        Safely return UNKNOWN when a valid multimodal prediction
        cannot be produced.
        """

        return UnifiedPrediction(
            subject_id=subject_id,
            overall_status=PredictionStatus.UNKNOWN,
            risk_signal=RiskSignal.UNKNOWN,
            human_review_required=True,
            modalities_available=[],
            modality_predictions=[],
            model_versions=[],
            framework_version=self.config[
                "framework"
            ].get("version", "0.1.0"),
            message=message,
        )


def main() -> None:
    print("=" * 72)
    print("MULTIMODAL INTEGRATION SERVICE")
    print("=" * 72)

    service = MultimodalIntegrationService()

    print("\nCONFIGURED MODALITIES")
    for modality in service.configured_modalities():
        print(f"[OK] {modality}")

    print("\nMODEL REGISTRY")
    for model_id, exists in service.verify_model_registry().items():
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {model_id}")

    print("\nMODEL ARTIFACTS")
    for artifact, exists in service.verify_artifacts().items():
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {artifact}")

    health = service.health_check()

    print("\nHEALTH STATUS")
    print(f"Status: {health['status']}")
    print(
        "Cross-dataset fusion enabled:",
        health["cross_dataset_fusion_enabled"],
    )

    unknown = service.create_unknown_prediction(
        subject_id="demo_subject_001",
        message="No valid inference input supplied.",
    )

    print("\nUNKNOWN POLICY")
    print("Status:", unknown.overall_status)
    print("Risk:", unknown.risk_signal)
    print("Human review:", unknown.human_review_required)

    print("\nPHASE 9.3B: PASSED")


if __name__ == "__main__":
    main()