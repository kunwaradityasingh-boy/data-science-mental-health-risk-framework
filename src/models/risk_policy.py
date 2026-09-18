from __future__ import annotations

from pathlib import Path

import yaml

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskLevel,
    ReviewRecommendation,
)


class RiskPolicyError(ValueError):
    """Raised when the risk policy is invalid."""


def load_risk_policy() -> dict:
    """
    Load the research risk-band policy.
    """

    base_dir = Path(__file__).resolve().parents[2]

    policy_path = (
        base_dir
        / "configs"
        / "risk_policy.yaml"
    )

    if not policy_path.exists():
        raise FileNotFoundError(
            f"Risk policy not found: {policy_path}"
        )

    with policy_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        policy = yaml.safe_load(file)

    if not isinstance(policy, dict):
        raise RiskPolicyError(
            "Risk policy must contain a YAML mapping."
        )

    return policy


def validate_score(score: float | None) -> None:
    """
    Validate model score.
    """

    if score is None:
        return

    if not isinstance(score, (int, float)):
        raise RiskPolicyError(
            "Model score must be numeric or None."
        )

    if not 0.0 <= float(score) <= 1.0:
        raise RiskPolicyError(
            "Model score must be between 0 and 1."
        )


def get_risk_level(
    score: float | None,
    data_quality: DataQualityStatus,
    policy: dict,
) -> RiskLevel:
    """
    Convert a model score into a research risk signal.

    UNKNOWN takes precedence when data quality is insufficient
    or when a model score is unavailable.
    """

    validate_score(score)

    unknown_conditions = policy.get(
        "unknown_conditions",
        {},
    )

    if score is None:
        if unknown_conditions.get(
            "missing_score",
            True,
        ):
            return RiskLevel.UNKNOWN

    if data_quality in {
        DataQualityStatus.INSUFFICIENT,
        DataQualityStatus.UNKNOWN,
    }:
        if unknown_conditions.get(
            "insufficient_data_quality",
            True,
        ):
            return RiskLevel.UNKNOWN

    if score is None:
        return RiskLevel.UNKNOWN

    bands = policy.get(
        "score_bands",
        {},
    )

    for level_name in (
        "low",
        "moderate",
        "high",
    ):

        band = bands.get(
            level_name
        )

        if not band:
            continue

        min_score = float(
            band["min_score"]
        )

        max_score = float(
            band["max_score"]
        )

        if (
            min_score
            <= float(score)
            <= max_score
        ):
            return RiskLevel(
                level_name
            )

    raise RiskPolicyError(
        f"No risk band matches score={score}."
    )


def get_review_recommendation(
    risk_level: RiskLevel,
    policy: dict,
) -> ReviewRecommendation:
    """
    Map research risk signal to human-review recommendation.
    """

    review_config = policy.get(
        "human_review",
        {},
    )

    mapping = {
        RiskLevel.LOW: (
            ReviewRecommendation.NOT_REQUIRED
        ),
        RiskLevel.MODERATE: (
            ReviewRecommendation.RECOMMENDED
        ),
        RiskLevel.HIGH: (
            ReviewRecommendation.REQUIRED
        ),
        RiskLevel.UNKNOWN: (
            ReviewRecommendation.UNAVAILABLE
        ),
    }

    configured_value = review_config.get(
        risk_level.value
    )

    if configured_value is None:
        return mapping[risk_level]

    return ReviewRecommendation(
        configured_value
    )