from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    """
    Standard risk categories used by the framework.

    These represent model screening categories, not medical diagnoses.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class LabelTask(str, Enum):
    """
    Describes what prediction task a dataset label represents.
    """

    BINARY_RISK = "binary_risk"
    MULTICLASS_RISK = "multiclass_risk"
    MENTAL_HEALTH_CATEGORY = "mental_health_category"
    EARLY_WARNING = "early_warning"
    UNKNOWN = "unknown"


class LabelDefinition(BaseModel):
    """
    Metadata describing how a target label should be interpreted.
    """

    model_config = ConfigDict(extra="forbid")

    original_label: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    normalized_risk: RiskLevel = RiskLevel.UNKNOWN

    task_type: LabelTask = LabelTask.UNKNOWN

    label_source: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    is_clinical_diagnosis: bool = False

    notes: str | None = Field(
        default=None,
        max_length=1000,
    )