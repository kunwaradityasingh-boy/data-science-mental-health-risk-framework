from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskOutput,
)
from src.features.dreaddit_features import (
    extract_engineered_features,
)
from src.models.risk_policy import (
    get_review_recommendation,
    get_risk_level,
    load_risk_policy,
)


class PredictionServiceError(RuntimeError):
    """Raised when model inference cannot be completed."""


class PredictionService:
    """
    Research prediction service for the current
    text + engineered-feature model.

    This service generates a research risk signal.
    It does not generate a clinical diagnosis.
    """

    def __init__(
        self,
        model_version: str = "text-engineered-logistic-v1",
    ) -> None:

        self.base_dir = (
            Path(__file__).resolve().parents[2]
        )

        self.model_dir = (
            self.base_dir
            / "models"
            / "combined_baseline"
        )

        self.vectorizer_path = (
            self.model_dir
            / "tfidf_vectorizer.joblib"
        )

        self.scaler_path = (
            self.model_dir
            / "engineered_scaler.joblib"
        )

        self.model_path = (
            self.model_dir
            / "logistic_regression.joblib"
        )

        if not self.vectorizer_path.exists():
            raise PredictionServiceError(
                f"TF-IDF vectorizer not found: "
                f"{self.vectorizer_path}"
            )

        if not self.scaler_path.exists():
            raise PredictionServiceError(
                f"Engineered scaler not found: "
                f"{self.scaler_path}"
            )

        if not self.model_path.exists():
            raise PredictionServiceError(
                f"Model not found: "
                f"{self.model_path}"
            )

        self.vectorizer = joblib.load(
            self.vectorizer_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        self.model = joblib.load(
            self.model_path
        )

        self.policy = load_risk_policy()

        self.model_version = model_version

    # =========================================================
    # TEXT PREDICTION
    # =========================================================

    def predict_text(
        self,
        text: str,
        subject_id: str,
        engineered_features: pd.DataFrame | None = None,
        available_modalities: list[str] | None = None,
        missing_modalities: list[str] | None = None,
        data_quality: DataQualityStatus = (
            DataQualityStatus.ACCEPTABLE
        ),
    ) -> RiskOutput:

        if not isinstance(text, str):
            raise PredictionServiceError(
                "text must be a string."
            )

        if not text.strip():
            raise PredictionServiceError(
                "text cannot be empty."
            )

        if not subject_id.strip():
            raise PredictionServiceError(
                "subject_id cannot be empty."
            )

        # -----------------------------------------------------
        # Current research model requires engineered features.
        # -----------------------------------------------------

        if engineered_features is None:
            raise PredictionServiceError(
                "The current combined model requires "
                "engineered features. "
                "Use the text-only model if only raw text "
                "is available."
            )

        if len(engineered_features) != 1:
            raise PredictionServiceError(
                "engineered_features must contain exactly "
                "one observation."
            )

        # -----------------------------------------------------
        # TF-IDF
        # -----------------------------------------------------

        text_matrix = self.vectorizer.transform(
            [text]
        )

        # -----------------------------------------------------
        # Engineered features
        # -----------------------------------------------------

        engineered_matrix = (
            extract_engineered_features(
                engineered_features
            )
        )

        scaled_engineered = (
            self.scaler.transform(
                engineered_matrix
            )
        )

        # -----------------------------------------------------
        # Combine
        # -----------------------------------------------------

        from scipy.sparse import csr_matrix, hstack

        combined_matrix = hstack(
            [
                text_matrix,
                csr_matrix(
                    scaled_engineered
                ),
            ],
            format="csr",
        )

        # -----------------------------------------------------
        # Prediction
        # -----------------------------------------------------

        predicted_label = int(
            self.model.predict(
                combined_matrix
            )[0]
        )

        class_1_score = float(
            self.model.predict_proba(
                combined_matrix
            )[0, 1]
        )

        # -----------------------------------------------------
        # Risk policy
        # -----------------------------------------------------

        risk_level = get_risk_level(
            score=class_1_score,
            data_quality=data_quality,
            policy=self.policy,
        )

        review_recommendation = (
            get_review_recommendation(
                risk_level=risk_level,
                policy=self.policy,
            )
        )

        # -----------------------------------------------------
        # Evidence
        # -----------------------------------------------------

        evidence = [
            "text-derived TF-IDF features",
            "engineered linguistic features",
        ]

        # -----------------------------------------------------
        # Modalities
        # -----------------------------------------------------

        if available_modalities is None:
            available_modalities = ["text"]

        if missing_modalities is None:
            missing_modalities = [
                "voice",
                "behavior",
            ]

        # -----------------------------------------------------
        # Structured output
        # -----------------------------------------------------

        return RiskOutput(
            prediction_id=(
                f"PRED-{uuid4().hex[:12]}"
            ),
            subject_id=subject_id,
            risk_level=risk_level,
            model_score=class_1_score,
            model_version=self.model_version,
            available_modalities=(
                available_modalities
            ),
            missing_modalities=(
                missing_modalities
            ),
            data_quality=data_quality,
            contributing_evidence=evidence,
            review_recommendation=(
                review_recommendation
            ),
            is_clinical_diagnosis=False,
        )