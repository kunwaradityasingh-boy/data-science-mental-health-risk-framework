from __future__ import annotations

from typing import Final

import pandas as pd


# Columns that identify the record/post rather than represent
# predictive features.
IDENTIFIER_COLUMNS: Final[set[str]] = {
    "id",
    "post_id",
    "subreddit",
    "text",
    "sentence_range",
}


# Target and annotation-related columns.
TARGET_COLUMNS: Final[set[str]] = {
    "label",
    "confidence",
}


def get_engineered_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return numeric engineered features available in Dreaddit.

    Identifier, raw text, target, and annotation columns are excluded.
    """

    excluded_columns = (
        IDENTIFIER_COLUMNS
        | TARGET_COLUMNS
    )

    feature_columns = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    return feature_columns


def extract_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the numeric engineered feature matrix.
    """

    feature_columns = get_engineered_feature_columns(df)

    if not feature_columns:
        raise ValueError(
            "No engineered numeric features were found."
        )

    features = df[feature_columns].copy()

    return features


def get_feature_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return a compact summary of engineered features.
    """

    feature_columns = get_engineered_feature_columns(df)

    return {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }