from enum import Enum


class MissingDataAction(str, Enum):
    IMPUTE = "impute"
    KEEP_MISSING = "keep_missing"
    EXCLUDE = "exclude"
    SKIP_MODALITY = "skip_modality"


def get_missing_data_action(
    field_type: str,
    is_training: bool,
) -> MissingDataAction:

    if field_type == "feature":
        return MissingDataAction.IMPUTE

    if field_type == "modality":
        return MissingDataAction.SKIP_MODALITY

    if field_type == "label":
        if is_training:
            return MissingDataAction.EXCLUDE

        return MissingDataAction.KEEP_MISSING

    return MissingDataAction.KEEP_MISSING
