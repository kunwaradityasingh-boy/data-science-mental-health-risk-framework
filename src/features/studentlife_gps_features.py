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
    / "gps.Rds"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "studentlife"
OUTPUT_FILE = OUTPUT_DIR / "gps_features.csv"


# GPS quality threshold.
# Locations with accuracy worse than 100 m are excluded
# from distance calculation, but retained for observation counts.
MAX_DISTANCE_ACCURACY_M = 100.0

# Maximum allowed time gap between consecutive GPS points
# used for distance calculation.
MAX_TIME_GAP_SECONDS = 30 * 60

# Maximum plausible movement speed used to reject
# obviously erroneous GPS jumps.
MAX_DISTANCE_SPEED_MPS = 50.0


def haversine_distance_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    earth_radius_m = 6_371_000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * earth_radius_m
        * np.arcsin(np.sqrt(a))
    )


def build_gps_features() -> pd.DataFrame:
    print(f"Reading: {INPUT_FILE}")

    result = pyreadr.read_r(str(INPUT_FILE))
    df = next(iter(result.values()))

    required = {
        "timestamp",
        "accuracy",
        "latitude",
        "longitude",
        "travelstate",
        "uid",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[
        [
            "timestamp",
            "accuracy",
            "latitude",
            "longitude",
            "travelstate",
            "uid",
        ]
    ].copy()

    df["uid"] = df["uid"].astype(str)

    numeric_columns = [
        "timestamp",
        "accuracy",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "accuracy",
            "latitude",
            "longitude",
            "uid",
        ]
    )

    # Valid geographic coordinate range.
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
    )

    df = df.dropna(
        subset=["datetime"]
    ).copy()

    df = df.sort_values(
        ["uid", "timestamp"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Basic participant-level observation features
    # ---------------------------------------------------------

    df["date"] = df["datetime"].dt.date

    observation_features = (
        df.groupby("uid")
        .agg(
            gps_observations=("uid", "size"),
            gps_observed_days=("date", "nunique"),
            gps_median_accuracy_m=("accuracy", "median"),
        )
    )

    observation_features[
        "gps_observations_per_day"
    ] = (
        observation_features["gps_observations"]
        / observation_features["gps_observed_days"]
    )

    # ---------------------------------------------------------
    # Travel-state features
    # ---------------------------------------------------------

    travel = (
        df["travelstate"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["is_moving"] = travel.eq("moving")
    df["is_stationary"] = travel.eq("stationary")

    travel_features = (
        df.groupby("uid")
        .agg(
            gps_moving_pct=("is_moving", "mean"),
            gps_stationary_pct=("is_stationary", "mean"),
        )
    )

    # ---------------------------------------------------------
    # Distance calculation
    # ---------------------------------------------------------

    # Only sufficiently accurate GPS points are used for distance.
    distance_df = df[
        df["accuracy"]
        <= MAX_DISTANCE_ACCURACY_M
    ].copy()

    distance_df = distance_df.sort_values(
        ["uid", "timestamp"]
    ).reset_index(drop=True)

    distance_df["prev_timestamp"] = (
        distance_df.groupby("uid")["timestamp"]
        .shift(1)
    )

    distance_df["prev_latitude"] = (
        distance_df.groupby("uid")["latitude"]
        .shift(1)
    )

    distance_df["prev_longitude"] = (
        distance_df.groupby("uid")["longitude"]
        .shift(1)
    )

    distance_df["time_gap_seconds"] = (
        distance_df["timestamp"]
        - distance_df["prev_timestamp"]
    )

    distance_df["segment_distance_m"] = (
        haversine_distance_m(
            distance_df["prev_latitude"],
            distance_df["prev_longitude"],
            distance_df["latitude"],
            distance_df["longitude"],
        )
    )

    # Remove first point of each participant and
    # implausible/too-large gaps.
    valid_segment = (
        distance_df["prev_timestamp"].notna()
        & distance_df["prev_latitude"].notna()
        & distance_df["prev_longitude"].notna()
        & (distance_df["time_gap_seconds"] > 0)
        & (
            distance_df["time_gap_seconds"]
            <= MAX_TIME_GAP_SECONDS
        )
        & distance_df["segment_distance_m"].notna()
    )

    # Calculate implied segment speed.
    distance_df["implied_speed_mps"] = (
        distance_df["segment_distance_m"]
        / distance_df["time_gap_seconds"]
    )

    valid_segment &= (
        distance_df["implied_speed_mps"]
        <= MAX_DISTANCE_SPEED_MPS
    )

    distance_df = distance_df[
        valid_segment
    ].copy()

    distance_df["distance_km"] = (
        distance_df["segment_distance_m"]
        / 1000.0
    )

    distance_df["date"] = (
        pd.to_datetime(
            distance_df["timestamp"],
            unit="s",
        ).dt.date
    )

    # Daily distance first.
    daily_distance = (
        distance_df.groupby(
            ["uid", "date"]
        )["distance_km"]
        .sum()
        .reset_index()
    )

    daily_features = (
        daily_distance.groupby("uid")
        .agg(
            gps_distance_total_km=(
                "distance_km",
                "sum",
            ),
            gps_distance_mean_daily_km=(
                "distance_km",
                "mean",
            ),
            gps_distance_median_daily_km=(
                "distance_km",
                "median",
            ),
            gps_distance_std_daily_km=(
                "distance_km",
                "std",
            ),
            gps_active_days=(
                "date",
                "nunique",
            ),
        )
    )

    # ---------------------------------------------------------
    # Combine all feature groups
    # ---------------------------------------------------------

    features = (
        observation_features
        .join(travel_features)
        .join(daily_features)
        .reset_index()
    )

    features = features.sort_values(
        "uid"
    ).reset_index(drop=True)

    # Replace numerical infinities.
    numeric_features = [
        column
        for column in features.columns
        if column != "uid"
    ]

    features[numeric_features] = (
        features[numeric_features]
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

    print("\nGPS feature extraction complete.")
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
    build_gps_features()