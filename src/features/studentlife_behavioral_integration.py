from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
)

OUTPUT_PATH = FEATURE_DIR / "behavioral_features.csv"


FEATURE_FILES = {
    "activity": FEATURE_DIR / "activity_features.csv",
    "phonelock": FEATURE_DIR / "phonelock_features.csv",
    "conversation": FEATURE_DIR / "conversation_features.csv",
    "gps": FEATURE_DIR / "gps_features.csv",
    "audio": FEATURE_DIR / "audio_features.csv",
}


def load_feature_file(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} feature file not found: {path}"
        )

    df = pd.read_csv(path)

    if "uid" not in df.columns:
        raise ValueError(
            f"{name} feature table does not contain 'uid'."
        )

    if df["uid"].duplicated().any():
        raise ValueError(
            f"{name} feature table contains duplicate UIDs."
        )

    if df["uid"].isna().any():
        raise ValueError(
            f"{name} feature table contains missing UIDs."
        )

    return df


def validate_feature_table(
    df: pd.DataFrame,
    expected_participants: int = 49,
) -> None:

    if df["uid"].duplicated().any():
        raise AssertionError("Duplicate UIDs found.")

    if df["uid"].isna().any():
        raise AssertionError("Missing UIDs found.")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    if df[numeric_columns].isna().any().any():
        raise AssertionError(
            "Missing numeric feature values found."
        )

    if np.isinf(df[numeric_columns].to_numpy()).any():
        raise AssertionError(
            "Infinite numeric feature values found."
        )

    if df["uid"].nunique() != expected_participants:
        raise AssertionError(
            f"Expected {expected_participants} participants, "
            f"found {df['uid'].nunique()}."
        )


def build_behavioral_features() -> pd.DataFrame:

    tables = {}

    for name, path in FEATURE_FILES.items():
        tables[name] = load_feature_file(name, path)

        print(
            f"{name:15s} -> "
            f"{tables[name].shape[0]} participants, "
            f"{tables[name].shape[1]} columns"
        )

    # Use activity as the reference participant set.
    merged = tables["activity"].copy()

    for name in [
        "phonelock",
        "conversation",
        "gps",
        "audio",
    ]:
        before = len(merged)

        merged = merged.merge(
            tables[name],
            on="uid",
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{name}"),
        )

        after = len(merged)

        print(
            f"Merged {name}: "
            f"{before} -> {after} participants"
        )

    merged = merged.sort_values("uid").reset_index(drop=True)

    return merged


def main() -> None:

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    behavioral = build_behavioral_features()

    validate_feature_table(behavioral)

    behavioral.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("========================================")
    print("StudentLife behavioral integration")
    print("========================================")
    print(f"Shape: {behavioral.shape}")
    print(f"Unique UIDs: {behavioral['uid'].nunique()}")
    print(f"Output: {OUTPUT_PATH}")
    print("Validation: PASSED")
    print("========================================")


if __name__ == "__main__":
    main()