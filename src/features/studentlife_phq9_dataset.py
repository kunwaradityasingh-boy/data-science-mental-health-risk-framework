from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEHAVIOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "behavioral_features.csv"
)

PHQ9_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "studentlife"
    / "extracted"
    / "dataset_rds"
    / "survey"
    / "PHQ-9.Rds"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "behavioral_phq9_dataset.csv"
)


PHQ9_MAPPING = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


def load_phq9():
    result = pyreadr.read_r(str(PHQ9_PATH))

    if not result:
        raise RuntimeError("Could not read PHQ-9.Rds")

    return next(iter(result.values()))


def calculate_phq9_score(df: pd.DataFrame) -> pd.DataFrame:

    question_columns = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7",
        "Q8",
        "Q9",
    ]

    missing_columns = set(question_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing PHQ-9 symptom columns: {sorted(missing_columns)}"
        )

    scored = df.copy()

    for column in question_columns:
        scored[column] = scored[column].map(PHQ9_MAPPING)

    if scored[question_columns].isna().any().any():
        raise ValueError(
            "Unexpected or missing PHQ-9 response encountered."
        )

    scored["phq9_score"] = scored[question_columns].sum(axis=1)

    return scored


def main():

    if not BEHAVIOR_PATH.exists():
        raise FileNotFoundError(BEHAVIOR_PATH)

    if not PHQ9_PATH.exists():
        raise FileNotFoundError(PHQ9_PATH)

    behavior = pd.read_csv(BEHAVIOR_PATH)

    if behavior["uid"].duplicated().any():
        raise ValueError("Behavioral dataset contains duplicate UIDs.")

    phq9 = load_phq9()

    # Keep the assessment identity.
    phq9["uid"] = phq9["uid"].astype(str)
    behavior["uid"] = behavior["uid"].astype(str)

    # Q10 is functional difficulty, NOT part of the PHQ-9 symptom score.
    phq9 = calculate_phq9_score(phq9)

    # Keep only research-relevant columns.
    phq9 = phq9[
        [
            "uid",
            "type",
            "phq9_score",
        ]
    ].copy()

    # Each UID may have pre and post assessments.
    # Preserve assessment type instead of pretending there is
    # a single timestamp-aligned target.
    merged = behavior.merge(
        phq9,
        on="uid",
        how="inner",
        validate="one_to_many",
    )

    merged = merged.sort_values(
        ["uid", "type"]
    ).reset_index(drop=True)

    # Validation
    if merged.empty:
        raise ValueError("No behavioral/PHQ-9 participant overlap.")

    if merged["phq9_score"].isna().any():
        raise ValueError("Missing PHQ-9 scores found.")

    if not merged["phq9_score"].between(0, 27).all():
        raise ValueError(
            "PHQ-9 score outside valid 0-27 range."
        )

    numeric_columns = merged.select_dtypes(
        include=[np.number]
    ).columns

    if np.isinf(merged[numeric_columns].to_numpy()).any():
        raise ValueError(
            "Infinite numeric values found."
        )

    # Important integrity checks
    common_uids = set(behavior["uid"]) & set(phq9["uid"])

    if merged["uid"].nunique() != len(common_uids):
        raise ValueError(
            "Unexpected participant intersection."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("========================================")
    print("StudentLife + PHQ-9 dataset")
    print("========================================")
    print(f"Behavioral participants: {behavior['uid'].nunique()}")
    print(f"PHQ-9 participants: {phq9['uid'].nunique()}")
    print(f"Common participants: {merged['uid'].nunique()}")
    print(f"Rows after pre/post preservation: {len(merged)}")
    print(f"PHQ-9 score range: {merged['phq9_score'].min()} - {merged['phq9_score'].max()}")
    print(f"PHQ-9 mean: {merged['phq9_score'].mean():.2f}")
    print()
    print("Assessment types:")
    print(merged["type"].value_counts().to_string())
    print()
    print(f"Output: {OUTPUT_PATH}")
    print("Validation: PASSED")
    print("========================================")


if __name__ == "__main__":
    main()