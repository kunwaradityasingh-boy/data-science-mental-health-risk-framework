from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "studentlife"
    / "extracted"
    / "dataset_rds"
    / "sensing"
    / "activity.Rds"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "studentlife"
OUTPUT_FILE = OUTPUT_DIR / "activity_features.csv"


ACTIVITY_MAP = {
    0.0: "stationary",
    1.0: "walking",
    2.0: "running",
    3.0: "unknown",
}


def build_activity_features() -> pd.DataFrame:
    print(f"Reading: {ACTIVITY_FILE}")

    result = pyreadr.read_r(str(ACTIVITY_FILE))
    df = next(iter(result.values()))

    required_columns = {"timestamp", "activity_inference", "uid"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["timestamp", "activity_inference", "uid"]].copy()

    # Normalize UID and activity code.
    df["uid"] = df["uid"].astype(str)
    df["activity_inference"] = pd.to_numeric(
        df["activity_inference"], errors="coerce"
    )

    # Remove invalid activity codes.
    df = df[df["activity_inference"].isin(ACTIVITY_MAP)].copy()

    # Convert Unix timestamp to date.
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
    )

    df = df.dropna(subset=["datetime", "uid"])

    # Total observations per participant.
    total = (
        df.groupby("uid")
        .size()
        .rename("activity_observations")
    )

    # Count each activity class.
    counts = pd.crosstab(
        df["uid"],
        df["activity_inference"],
    )

    counts = counts.rename(
        columns={
            0.0: "stationary_count",
            1.0: "walking_count",
            2.0: "running_count",
            3.0: "unknown_count",
        }
    )

    # Ensure every expected column exists.
    for column in [
        "stationary_count",
        "walking_count",
        "running_count",
        "unknown_count",
    ]:
        if column not in counts.columns:
            counts[column] = 0

    counts = counts[
        [
            "stationary_count",
            "walking_count",
            "running_count",
            "unknown_count",
        ]
    ]

    features = counts.join(total)

    # Activity proportions.
    features["stationary_pct"] = (
        features["stationary_count"]
        / features["activity_observations"]
    )

    features["walking_pct"] = (
        features["walking_count"]
        / features["activity_observations"]
    )

    features["running_pct"] = (
        features["running_count"]
        / features["activity_observations"]
    )

    features["unknown_pct"] = (
        features["unknown_count"]
        / features["activity_observations"]
    )

    # Non-stationary activity = walking + running.
    features["nonstationary_pct"] = (
        features["walking_count"]
        + features["running_count"]
    ) / features["activity_observations"]

    # Number of calendar days with recorded activity.
    daily_counts = (
        df.assign(date=df["datetime"].dt.date)
        .groupby("uid")["date"]
        .nunique()
        .rename("observed_days")
    )

    features = features.join(daily_counts)

    # Average number of observations per observed day.
    features["observations_per_day"] = (
        features["activity_observations"]
        / features["observed_days"]
    )

    # Entropy of activity distribution.
    def entropy(row):
        probabilities = np.array(
            [
                row["stationary_pct"],
                row["walking_pct"],
                row["running_pct"],
                row["unknown_pct"],
            ],
            dtype=float,
        )

        probabilities = probabilities[probabilities > 0]

        if len(probabilities) == 0:
            return 0.0

        return float(
            -np.sum(probabilities * np.log2(probabilities))
        )

    features["activity_entropy"] = features.apply(
        entropy,
        axis=1,
    )

    # Put UID back as a normal column.
    features = features.reset_index()

    # Sort for reproducibility.
    features = features.sort_values("uid").reset_index(drop=True)

    # Numeric cleanup.
    numeric_columns = [
        column
        for column in features.columns
        if column != "uid"
    ]

    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nActivity feature extraction complete.")
    print(f"Participants: {len(features)}")
    print(f"Features: {len(features.columns) - 1}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFeature columns:")
    for column in features.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(features.head(10).to_string(index=False))

    return features


if __name__ == "__main__":
    build_activity_features()