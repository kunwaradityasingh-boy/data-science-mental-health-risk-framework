from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "studentlife"
    / "extracted"
    / "dataset_rds"
    / "sensing"
    / "conversation.Rds"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "studentlife"
OUTPUT_FILE = OUTPUT_DIR / "conversation_features.csv"


def build_conversation_features() -> pd.DataFrame:
    print(f"Reading: {INPUT_FILE}")

    result = pyreadr.read_r(str(INPUT_FILE))
    df = next(iter(result.values()))

    required = {
        "start_timestamp",
        "end_timestamp",
        "uid",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[
        ["start_timestamp", "end_timestamp", "uid"]
    ].copy()

    df["uid"] = df["uid"].astype(str)

    df["start_timestamp"] = pd.to_numeric(
        df["start_timestamp"],
        errors="coerce",
    )

    df["end_timestamp"] = pd.to_numeric(
        df["end_timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "start_timestamp",
            "end_timestamp",
            "uid",
        ]
    )

    df["start_datetime"] = pd.to_datetime(
        df["start_timestamp"],
        unit="s",
        errors="coerce",
    )

    df["end_datetime"] = pd.to_datetime(
        df["end_timestamp"],
        unit="s",
        errors="coerce",
    )

    df["duration_seconds"] = (
        df["end_timestamp"]
        - df["start_timestamp"]
    )

    invalid = (
        df["duration_seconds"].isna()
        | (df["duration_seconds"] <= 0)
    )

    invalid_count = int(invalid.sum())

    if invalid_count:
        print(
            f"Removing {invalid_count} invalid intervals."
        )

    df = df.loc[~invalid].copy()

    df["duration_hours"] = (
        df["duration_seconds"] / 3600.0
    )

    df["date"] = df["start_datetime"].dt.date

    grouped = df.groupby("uid")

    features = grouped["duration_seconds"].agg(
        conversation_events="count",
        conversation_total_duration_seconds="sum",
        conversation_mean_duration_seconds="mean",
        conversation_median_duration_seconds="median",
        conversation_std_duration_seconds="std",
        conversation_min_duration_seconds="min",
        conversation_max_duration_seconds="max",
    )

    observed_days = (
        grouped["date"]
        .nunique()
        .rename("conversation_observed_days")
    )

    features = features.join(observed_days)

    duration_columns = [
        "conversation_total_duration_seconds",
        "conversation_mean_duration_seconds",
        "conversation_median_duration_seconds",
        "conversation_std_duration_seconds",
        "conversation_min_duration_seconds",
        "conversation_max_duration_seconds",
    ]

    for column in duration_columns:
        output_column = column.replace(
            "_seconds",
            "_hours",
        )

        features[output_column] = (
            features[column] / 3600.0
        )

    features = features.drop(
        columns=duration_columns
    )

    features["conversation_events_per_day"] = (
        features["conversation_events"]
        / features["conversation_observed_days"]
    )

    features["conversation_duration_hours_per_day"] = (
        features["conversation_total_duration_hours"]
        / features["conversation_observed_days"]
    )

    features = features.reset_index()

    features = features.sort_values(
        "uid"
    ).reset_index(drop=True)

    numeric_columns = [
        c for c in features.columns
        if c != "uid"
    ]

    features[numeric_columns] = (
        features[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nConversation feature extraction complete.")
    print(f"Participants: {len(features)}")
    print(f"Features: {len(features.columns) - 1}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFeature columns:")
    for column in features.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(
        features.head(10).to_string(index=False)
    )

    return features


if __name__ == "__main__":
    build_conversation_features()