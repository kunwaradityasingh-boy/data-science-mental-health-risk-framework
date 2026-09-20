from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "studentlife"
PHQ_FILE = DATA_DIR / "phq9_scores.csv"

OUTPUT_FILE = DATA_DIR / "behavior_fused_pre.csv"


FEATURE_FILES = [
    "activity_features.csv",
    "phonelock_features.csv",
    "conversation_features.csv",
    "gps_features.csv",
    "audio_features.csv",
]


# ============================================================
# HELPERS
# ============================================================

def load_feature_file(filename: str) -> pd.DataFrame:
    """
    Load a participant-level behavioral feature file.

    Every UID is explicitly converted to string so that all
    feature tables use the same identifier type.
    """

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {path}"
        )

    df = pd.read_csv(path)

    if "uid" not in df.columns:
        raise ValueError(
            f"'uid' column missing from {filename}"
        )

    # Standardize UID type
    df["uid"] = df["uid"].astype(str).str.strip()

    # UID must be unique in participant-level feature files
    if df["uid"].duplicated().any():
        duplicates = df.loc[
            df["uid"].duplicated(),
            "uid"
        ].unique()

        raise ValueError(
            f"Duplicate UIDs found in {filename}: "
            f"{duplicates[:10]}"
        )

    return df


def validate_numeric_features(df: pd.DataFrame) -> None:
    """
    Validate all feature columns.
    """

    feature_columns = [
        c for c in df.columns
        if c != "uid"
    ]

    if not feature_columns:
        raise ValueError(
            "No behavioral feature columns found."
        )

    numeric_df = df[feature_columns]

    # Check numeric conversion
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(
            numeric_df[column]
        ):
            raise ValueError(
                f"Non-numeric behavioral feature: {column}"
            )

    # Missing values
    missing_count = int(
        numeric_df.isna().sum().sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Behavioral features contain "
            f"{missing_count} missing values."
        )

    # Infinite values
    infinite_count = int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

    if infinite_count != 0:
        raise ValueError(
            f"Behavioral features contain "
            f"{infinite_count} infinite values."
        )


# ============================================================
# LOAD BEHAVIORAL FEATURES
# ============================================================

behavioral_tables = []

for filename in FEATURE_FILES:

    df = load_feature_file(filename)

    print(
        f"{filename}: "
        f"shape={df.shape}, "
        f"uids={df['uid'].nunique()}"
    )

    behavioral_tables.append(df)


# ============================================================
# MERGE BEHAVIORAL FEATURES
# ============================================================

fused = behavioral_tables[0].copy()

for df in behavioral_tables[1:]:

    fused = fused.merge(
        df,
        on="uid",
        how="inner",
        validate="one_to_one",
    )


print(
    f"After behavioral merge: {fused.shape}"
)

print(
    f"Behavioral UIDs: "
    f"{fused['uid'].nunique()}"
)


# ============================================================
# LOAD PHQ-9
# ============================================================

if not PHQ_FILE.exists():
    raise FileNotFoundError(
        f"Missing PHQ-9 file: {PHQ_FILE}"
    )

phq = pd.read_csv(PHQ_FILE)

required_phq_columns = {
    "uid",
    "type",
    "phq9_score",
}

missing_phq_columns = (
    required_phq_columns -
    set(phq.columns)
)

if missing_phq_columns:
    raise ValueError(
        "PHQ-9 file is missing columns: "
        f"{sorted(missing_phq_columns)}"
    )


# Standardize UID
phq["uid"] = (
    phq["uid"]
    .astype(str)
    .str.strip()
)


# Normalize assessment type
phq["type"] = (
    phq["type"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# KEEP ONLY PRE ASSESSMENT
# ============================================================
#
# IMPORTANT:
# We intentionally use the PRE assessment only.
#
# The current StudentLife PHQ-9 table does not provide an
# exact assessment timestamp. Therefore we cannot honestly
# construct a temporally aligned "early prediction" dataset
# from the whole-study behavioral aggregates.
#
# This table is therefore a participant-level retrospective
# association baseline, not a temporal prediction dataset.
# ============================================================

phq_pre = phq[
    phq["type"] == "pre"
].copy()


if phq_pre.empty:
    raise ValueError(
        "No PRE PHQ-9 records found."
    )


# PRE should contain at most one record per UID
if phq_pre["uid"].duplicated().any():

    duplicates = phq_pre.loc[
        phq_pre["uid"].duplicated(),
        "uid"
    ].unique()

    raise ValueError(
        "Multiple PRE PHQ-9 records found for "
        f"UIDs: {duplicates[:10]}"
    )


# ============================================================
# MERGE PRE PHQ-9 WITH BEHAVIORAL FEATURES
# ============================================================

fused["uid"] = fused["uid"].astype(str)

fused = fused.merge(
    phq_pre[
        [
            "uid",
            "phq9_score",
        ]
    ],
    on="uid",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# FINAL DATASET METADATA
# ============================================================

fused["target_source"] = "StudentLife PHQ-9 PRE"
fused["target_definition"] = (
    "PHQ-9 symptom score from pre assessment"
)

fused["dataset_role"] = (
    "retrospective participant-level "
    "behavioral association baseline"
)

fused["temporal_alignment"] = (
    "not available from source PHQ-9 table"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print(
    f"Rows: {len(fused)}"
)

print(
    f"Unique UIDs: "
    f"{fused['uid'].nunique()}"
)

print(
    f"PHQ-9 PRE records: "
    f"{len(phq_pre)}"
)

print(
    f"PHQ-9 score range: "
    f"{fused['phq9_score'].min()} "
    f"to "
    f"{fused['phq9_score'].max()}"
)


# UID uniqueness
if fused["uid"].duplicated().any():
    raise ValueError(
        "Final dataset contains duplicate UIDs."
    )


# Numeric feature validation
validate_numeric_features(
    fused.drop(
        columns=[
            "target_source",
            "target_definition",
            "dataset_role",
            "temporal_alignment",
        ],
        errors="ignore",
    )
)


# Target validation
if fused["phq9_score"].isna().any():
    raise ValueError(
        "Missing PHQ-9 target values."
    )

if not pd.api.types.is_numeric_dtype(
    fused["phq9_score"]
):
    raise ValueError(
        "PHQ-9 score is not numeric."
    )


if np.isinf(
    fused["phq9_score"].to_numpy()
).any():
    raise ValueError(
        "Infinite PHQ-9 target values found."
    )


# Expected PHQ-9 range
if (
    fused["phq9_score"].min() < 0
    or fused["phq9_score"].max() > 27
):
    raise ValueError(
        "PHQ-9 score outside valid 0-27 range."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fused.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    f"Behavioral feature columns: "
    f"{len(behavioral_tables[0].columns) - 1}"
    f" + merged modalities"
)

print(
    f"Final columns: {fused.shape[1]}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print()
print("Validation: PASSED")
print("=" * 60)