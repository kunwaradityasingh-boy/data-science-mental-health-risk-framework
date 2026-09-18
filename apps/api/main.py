from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskOutput,
)
from src.models.prediction_service import (
    PredictionService,
    PredictionServiceError,
)


app = FastAPI(
    title="Mental Health Risk Research API",
    description=(
        "Research-only API for generating model-based "
        "mental-health risk signals. "
        "This system is not a clinical diagnostic system."
    ),
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    subject_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
    )

    engineered_features: dict[str, float] = Field(
        ...,
        description=(
            "The 109 engineered features required by "
            "the current combined research model."
        ),
    )

    available_modalities: list[str] = Field(
        default_factory=lambda: ["text"],
    )

    missing_modalities: list[str] = Field(
        default_factory=lambda: [
            "voice",
            "behavior",
        ],
    )

    data_quality: DataQualityStatus = (
        DataQualityStatus.ACCEPTABLE
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ------------------------------------------------------------
# Load model once when API starts.
# ------------------------------------------------------------

try:
    prediction_service = PredictionService()
    model_load_error: str | None = None

except Exception as exc:
    prediction_service = None
    model_load_error = str(exc)


@app.get(
    "/",
    response_model=HealthResponse,
)
def root() -> HealthResponse:

    return HealthResponse(
        status="ok",
        service="mental-health-risk-research-api",
        version="0.1.0",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    if prediction_service is None:

        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "reason": model_load_error,
            },
        )

    return HealthResponse(
        status="healthy",
        service="mental-health-risk-research-api",
        version="0.1.0",
    )


@app.post(
    "/predict",
    response_model=RiskOutput,
)
def predict(
    request: PredictionRequest,
) -> RiskOutput:

    if prediction_service is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is unavailable."
            ),
        )

    # --------------------------------------------------------
    # Convert engineered feature dictionary to DataFrame.
    # --------------------------------------------------------

    engineered_df = pd.DataFrame(
        [request.engineered_features]
    )

    # --------------------------------------------------------
    # Validate that exactly the expected feature count
    # is present.
    # --------------------------------------------------------

    try:

        result = prediction_service.predict_text(
            text=request.text,
            subject_id=request.subject_id,
            engineered_features=engineered_df,
            available_modalities=(
                request.available_modalities
            ),
            missing_modalities=(
                request.missing_modalities
            ),
            data_quality=request.data_quality,
        )

        return result

    except PredictionServiceError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed."
            ),
        ) from exc