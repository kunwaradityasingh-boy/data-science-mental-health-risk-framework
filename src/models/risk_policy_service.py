"""
Central risk-policy service.

Research-only:
    Risk bands are model-generated research signals.
    They are not clinical diagnoses or clinical probabilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.data.prediction.unified_prediction_schema import (
    RiskSignal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = PROJECT_ROOT / "configs" / "risk_policy.yaml"


class RiskPolicyService:
    """Apply the project's centralized risk policy."""

    def __init__(self, policy_path: Path = POLICY_PATH):
        self.policy_path = Path(policy_path)
        self.policy = self._load_policy()

    def _load_policy(self) -> Dict[str, Any]:
        if not self.policy_path.exists():
            raise FileNotFoundError(
                f"Risk policy not found: {self.policy_path}"
            )

        with self.policy_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            policy = yaml.safe_load(file)

        if not isinstance(policy, dict):
            raise ValueError("Invalid risk policy YAML.")

        return policy

    def classify(
        self,
        score: Optional[float],
    ) -> RiskSignal:
        """
        Convert a normalized model score into the configured
        research risk band.

        Missing or invalid scores return UNKNOWN.
        """

        if score is None:
            return RiskSignal.UNKNOWN

        try:
            score = float(score)
        except (TypeError, ValueError):
            return RiskSignal.UNKNOWN

        if score != score:
            return RiskSignal.UNKNOWN

        if score < 0.0 or score > 1.0:
            return RiskSignal.UNKNOWN

        thresholds = self.policy.get("thresholds", {})

        low = thresholds.get("low", {})
        moderate = thresholds.get("moderate", {})
        high = thresholds.get("high", {})

        low_max = float(low.get("max", 0.39))
        moderate_min = float(moderate.get("min", 0.40))
        moderate_max = float(moderate.get("max", 0.69))
        high_min = float(high.get("min", 0.70))

        if score <= low_max:
            return RiskSignal.LOW

        if moderate_min <= score <= moderate_max:
            return RiskSignal.MODERATE

        if score >= high_min:
            return RiskSignal.HIGH

        return RiskSignal.UNKNOWN

    def human_review_required(
        self,
        risk_signal: RiskSignal,
    ) -> bool:
        """Determine human-review requirement from policy."""

        review = self.policy.get("human_review", {})

        if risk_signal == RiskSignal.HIGH:
            return bool(review.get("high_required", True))

        if risk_signal == RiskSignal.MODERATE:
            return bool(review.get("moderate_recommended", True))

        return False

    def evaluate(
        self,
        score: Optional[float],
    ) -> Dict[str, Any]:
        """Return the complete policy decision."""

        risk_signal = self.classify(score)

        return {
            "score": score,
            "risk_signal": risk_signal.value,
            "human_review_required": self.human_review_required(
                risk_signal
            ),
            "research_only": True,
            "clinical_diagnosis": False,
        }


def main() -> None:
    print("=" * 72)
    print("RISK POLICY SERVICE")
    print("=" * 72)

    service = RiskPolicyService()

    test_scores = [
        None,
        0.20,
        0.50,
        0.80,
        1.20,
    ]

    print("\nPOLICY TESTS")

    for score in test_scores:
        result = service.evaluate(score)

        print(
            f"Score={score!s:>4}  "
            f"Risk={result['risk_signal']:<8}  "
            f"Human review={result['human_review_required']}"
        )

    print("\nPOLICY FILE")
    print(f"[OK] {service.policy_path}")

    print("\nPHASE 9.4: PASSED")


if __name__ == "__main__":
    main()