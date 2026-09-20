"""
Framework FastAPI application.

Phase 9.6
Connects the framework prediction service to FastAPI.

Research-only API.
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.models.framework_prediction_service import (
    FrameworkPredictionService,
)


app = FastAPI(
    title="Mental Health Risk Research Framework API",
    description=(
        "Research-only API for the Data Science Framework "
        "for Early Detection of Mental Health Risk."
    ),
    version="0.1.0",
)


service = FrameworkPredictionService()


# ============================================================
# REQUEST SCHEMA
# ============================================================

class FrameworkPredictionRequest(BaseModel):
    subject_id: str = Field(
        ...,
        min_length=1,
        description="Pseudonymous subject identifier.",
    )

    text_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    voice_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    behavior_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Mental Health Risk Research Framework API",
        "version": "0.1.0",
        "research_only": True,
        "clinical_diagnosis": False,
        "message": (
            "Research-only multimodal risk framework."
        ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return service.health()


# ============================================================
# FRAMEWORK PREDICTION
# ============================================================

@app.post("/framework/predict")
def framework_predict(
    request: FrameworkPredictionRequest,
):
    result = service.predict(
        subject_id=request.subject_id,
        text_score=request.text_score,
        voice_score=request.voice_score,
        behavior_score=request.behavior_score,
    )

    return result.model_dump(mode="json")