import pytest

from src.data.risk_output_schema import (
    DataQualityStatus,
    RiskLevel,
    ReviewRecommendation,
)

from src.models.risk_policy import (
    get_review_recommendation,
    get_risk_level,
    load_risk_policy,
)


@pytest.fixture
def policy():
    return load_risk_policy()


def test_low_score(policy):

    result = get_risk_level(
        score=0.20,
        data_quality=(
            DataQualityStatus.ACCEPTABLE
        ),
        policy=policy,
    )

    assert result == RiskLevel.LOW


def test_moderate_score(policy):

    result = get_risk_level(
        score=0.55,
        data_quality=(
            DataQualityStatus.ACCEPTABLE
        ),
        policy=policy,
    )

    assert result == RiskLevel.MODERATE


def test_high_score(policy):

    result = get_risk_level(
        score=0.85,
        data_quality=(
            DataQualityStatus.ACCEPTABLE
        ),
        policy=policy,
    )

    assert result == RiskLevel.HIGH


def test_missing_score_returns_unknown(policy):

    result = get_risk_level(
        score=None,
        data_quality=(
            DataQualityStatus.ACCEPTABLE
        ),
        policy=policy,
    )

    assert result == RiskLevel.UNKNOWN


def test_insufficient_data_returns_unknown(policy):

    result = get_risk_level(
        score=0.90,
        data_quality=(
            DataQualityStatus.INSUFFICIENT
        ),
        policy=policy,
    )

    assert result == RiskLevel.UNKNOWN


def test_unknown_data_quality_returns_unknown(policy):

    result = get_risk_level(
        score=0.90,
        data_quality=(
            DataQualityStatus.UNKNOWN
        ),
        policy=policy,
    )

    assert result == RiskLevel.UNKNOWN


def test_invalid_score_rejected(policy):

    with pytest.raises(ValueError):

        get_risk_level(
            score=1.5,
            data_quality=(
                DataQualityStatus.ACCEPTABLE
            ),
            policy=policy,
        )


def test_review_recommendation(policy):

    assert (
        get_review_recommendation(
            RiskLevel.LOW,
            policy,
        )
        == ReviewRecommendation.NOT_REQUIRED
    )

    assert (
        get_review_recommendation(
            RiskLevel.MODERATE,
            policy,
        )
        == ReviewRecommendation.RECOMMENDED
    )

    assert (
        get_review_recommendation(
            RiskLevel.HIGH,
            policy,
        )
        == ReviewRecommendation.REQUIRED
    )

    assert (
        get_review_recommendation(
            RiskLevel.UNKNOWN,
            policy,
        )
        == ReviewRecommendation.UNAVAILABLE
    )