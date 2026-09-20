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
    / "survey"
    / "PHQ-9.Rds"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "phq9_scores.csv"
)


SCORE_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


def load_phq9() -> pd.DataFrame:
    result = pyreadr.read_r(str(INPUT_FILE))

    if not result:
        raise RuntimeError("Unable to read PHQ-9 RDS file.")

    return next(iter(result.values()))


def score_phq9(df: pd.DataFrame) -> pd.DataFrame:
    question_columns = [f"Q{i}" for i in range(1, 10)]

    scored = df.copy()

    for column in question_columns:
        scored[column] = scored[column].astype(str).map(SCORE_MAP)

        if scored[column].isna().any():
            raise ValueError(
                f"Unknown or missing response found in {column}."
            )

    scored["phq9_score"] = scored[question_columns].sum(axis=1)

    return scored[
        ["uid", "type", "phq9_score"]
    ]


def validate_scores(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("PHQ-9 score table is empty.")

    if df["uid"].isna().any():
        raise ValueError("Missing UID detected.")

    if df["phq9_score"].isna().any():
        raise ValueError("Missing PHQ-9 score detected.")

    if not df["phq9_score"].between(0, 27).all():
        raise ValueError("PHQ-9 score outside valid 0–27 range.")

    if df.duplicated(["uid", "type"]).any():
        raise ValueError(
            "Duplicate participant/timepoint detected."
        )

    if not set(df["type"].unique()).issubset({"pre", "post"}):
        raise ValueError("Unexpected assessment type detected.")


def main() -> None:
    raw = load_phq9()

    scored = score_phq9(raw)

    validate_scores(scored)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    scored.to_csv(OUTPUT_FILE, index=False)

    print("PHQ-9 scoring complete.")
    print(f"Shape: {scored.shape}")
    print(f"Unique UIDs: {scored['uid'].nunique()}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nScore range:")
    print(
        scored["phq9_score"].min(),
        "to",
        scored["phq9_score"].max(),
    )

    print("\nAssessment counts:")
    print(scored["type"].value_counts())

    print("\nValidation: PASSED")


if __name__ == "__main__":
    main()