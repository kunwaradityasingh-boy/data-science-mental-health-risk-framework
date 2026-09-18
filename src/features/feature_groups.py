from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.features.dreaddit_features import (
    get_engineered_feature_columns,
)


FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "social": [
        "social_timestamp",
        "social_karma",
        "social_upvote_ratio",
        "social_num_comments",
    ],
    "syntax_readability": [
        "syntax_ari",
        "syntax_fk_grade",
    ],
    "sentiment": [
        "sentiment",
    ],
}


def classify_feature_group(column: str) -> str:
    """
    Classify an engineered Dreaddit feature
    into its research feature group.
    """

    if column.startswith("lex_liwc_"):
        return "liwc"

    if column.startswith("lex_dal_"):
        return "dal"

    for group_name, columns in FEATURE_GROUPS.items():
        if column in columns:
            return group_name

    return "other"


def group_features(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Group ONLY the engineered numeric features.

    Identifier, raw text, target, and annotation columns
    are excluded by get_engineered_feature_columns().
    """

    engineered_columns = get_engineered_feature_columns(df)

    grouped: dict[str, list[str]] = {}

    for column in engineered_columns:
        group = classify_feature_group(column)

        grouped.setdefault(group, [])
        grouped[group].append(column)

    return grouped


def print_feature_group_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print the number of engineered features in each group.
    """

    grouped = group_features(df)

    print("=" * 60)
    print("DREAddit ENGINEERED FEATURE GROUP SUMMARY")
    print("=" * 60)

    total = 0

    for group_name in sorted(grouped):
        count = len(grouped[group_name])

        print(
            f"{group_name:20s}: {count}"
        )

        total += count

    print("-" * 60)

    print(
        f"{'Total engineered features':20s}: {total}"
    )

    print("=" * 60)


def validate_feature_groups(
    df: pd.DataFrame,
) -> None:
    """
    Validate that every engineered feature belongs to
    exactly one recognized feature group.
    """

    engineered_columns = set(
        get_engineered_feature_columns(df)
    )

    grouped = group_features(df)

    grouped_columns: list[str] = []

    for columns in grouped.values():
        grouped_columns.extend(columns)

    grouped_columns_set = set(grouped_columns)

    duplicate_columns = {
        column
        for column in grouped_columns
        if grouped_columns.count(column) > 1
    }

    missing_columns = (
        engineered_columns - grouped_columns_set
    )

    unexpected_columns = (
        grouped_columns_set - engineered_columns
    )

    if duplicate_columns:
        raise ValueError(
            f"Features assigned to multiple groups: "
            f"{sorted(duplicate_columns)}"
        )

    if missing_columns:
        raise ValueError(
            f"Engineered features without a group: "
            f"{sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Non-engineered columns found in groups: "
            f"{sorted(unexpected_columns)}"
        )

    print(
        "[OK] Every engineered feature belongs to exactly one group."
    )

    print(
        "[OK] No duplicate feature-group assignments."
    )

    print(
        "[OK] No missing engineered features."
    )

    print(
        "[OK] No non-engineered columns included."
    )


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]

    dataset_path = (
        BASE_DIR
        / "data"
        / "raw"
        / "dreaddit"
        / "extracted"
        / "dreaddit-train.csv"
    )

    df = pd.read_csv(dataset_path)

    print_feature_group_summary(df)

    print()

    validate_feature_groups(df)