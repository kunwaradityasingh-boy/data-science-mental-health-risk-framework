from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# EATD VOICE ERROR + THRESHOLD ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
    / "acoustic_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "eatd_voice"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)


required_columns = {
    "split",
    "participant",
    "new_label",
}

missing = (
    required_columns
    - set(df.columns)
)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


validation_df = df[
    df["split"].astype(str).str.lower()
    == "validation"
].copy()


validation_df["new_label"] = pd.to_numeric(
    validation_df["new_label"],
    errors="coerce",
)

validation_df["target"] = (
    validation_df["new_label"] > 52.0
).astype(int)


# ============================================================
# FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {
    "split",
    "participant",
    "label",
    "new_label",
    "target",
    "usable_responses",
}

feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


X_validation = validation_df[
    feature_columns
].copy()

X_validation = X_validation.apply(
    pd.to_numeric,
    errors="coerce",
)

y_validation = validation_df[
    "target"
].astype(int)


# ============================================================
# LOAD MODELS
# ============================================================

logistic_file = (
    MODEL_DIR
    / "logistic_regression.joblib"
)

random_forest_file = (
    MODEL_DIR
    / "random_forest.joblib"
)


if not logistic_file.exists():
    raise FileNotFoundError(
        f"Missing model:\n{logistic_file}"
    )

if not random_forest_file.exists():
    raise FileNotFoundError(
        f"Missing model:\n{random_forest_file}"
    )


logistic_model = joblib.load(
    logistic_file
)

random_forest_model = joblib.load(
    random_forest_file
)


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

logistic_probability = (
    logistic_model.predict_proba(
        X_validation
    )[:, 1]
)

random_forest_probability = (
    random_forest_model.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# BASIC MODEL ANALYSIS
# ============================================================

print("=" * 70)
print("EATD VOICE ERROR + THRESHOLD ANALYSIS")
print("=" * 70)

print()
print(
    f"Validation participants: "
    f"{len(validation_df)}"
)

print(
    f"Depressed cases: "
    f"{int(y_validation.sum())}"
)

print(
    f"Non-depressed cases: "
    f"{int((y_validation == 0).sum())}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    model_name,
    probabilities,
):

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_validation,
            predictions,
        )

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
            }
        )

    return rows


threshold_rows = []

threshold_rows.extend(
    threshold_analysis(
        "Logistic Regression",
        logistic_probability,
    )
)

threshold_rows.extend(
    threshold_analysis(
        "Random Forest",
        random_forest_probability,
    )
)


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# PRINT THRESHOLD RESULTS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_file = (
    REPORT_DIR
    / "eatd_voice_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False,
)


# ============================================================
# PROBABILITY SUMMARY
# ============================================================

probability_summary = pd.DataFrame(
    {
        "model": [
            "Logistic Regression",
            "Random Forest",
        ],
        "min_probability": [
            logistic_probability.min(),
            random_forest_probability.min(),
        ],
        "mean_probability": [
            logistic_probability.mean(),
            random_forest_probability.mean(),
        ],
        "median_probability": [
            np.median(logistic_probability),
            np.median(random_forest_probability),
        ],
        "max_probability": [
            logistic_probability.max(),
            random_forest_probability.max(),
        ],
        "roc_auc": [
            roc_auc_score(
                y_validation,
                logistic_probability,
            ),
            roc_auc_score(
                y_validation,
                random_forest_probability,
            ),
        ],
        "pr_auc": [
            average_precision_score(
                y_validation,
                logistic_probability,
            ),
            average_precision_score(
                y_validation,
                random_forest_probability,
            ),
        ],
    }
)


print()
print("=" * 70)
print("PROBABILITY SUMMARY")
print("=" * 70)

print(
    probability_summary.to_string(
        index=False
    )
)


probability_file = (
    REPORT_DIR
    / "eatd_voice_probability_summary.csv"
)

probability_summary.to_csv(
    probability_file,
    index=False,
)


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

false_negative_rows = []


for model_name, probabilities in [
    (
        "Logistic Regression",
        logistic_probability,
    ),
    (
        "Random Forest",
        random_forest_probability,
    ),
]:

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    false_negative_mask = (
        (y_validation == 1)
        & (predictions == 0)
    )

    for index in validation_df[
        false_negative_mask
    ].index:

        false_negative_rows.append(
            {
                "model": model_name,
                "participant": validation_df.loc[
                    index,
                    "participant",
                ],
                "new_label": validation_df.loc[
                    index,
                    "new_label",
                ],
                "probability": (
                    probabilities[
                        list(validation_df.index)
                        .index(index)
                    ]
                ),
            }
        )


false_negative_df = pd.DataFrame(
    false_negative_rows
)


false_negative_file = (
    REPORT_DIR
    / "eatd_voice_false_negatives.csv"
)

false_negative_df.to_csv(
    false_negative_file,
    index=False,
)


# ============================================================
# COMPLETE REPORT
# ============================================================

report_file = (
    REPORT_DIR
    / "eatd_voice_error_analysis_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "EATD Voice Error and Threshold Analysis\n"
    )

    report.write(
        "=======================================\n\n"
    )

    report.write(
        f"Validation participants: "
        f"{len(validation_df)}\n"
    )

    report.write(
        f"Depressed cases: "
        f"{int(y_validation.sum())}\n"
    )

    report.write(
        f"Non-depressed cases: "
        f"{int((y_validation == 0).sum())}\n\n"
    )

    report.write(
        "Threshold analysis:\n"
    )

    report.write(
        threshold_df.to_string(
            index=False
        )
    )

    report.write(
        "\n\nProbability summary:\n"
    )

    report.write(
        probability_summary.to_string(
            index=False
        )
    )

    report.write(
        "\n\nFalse-negative analysis uses "
        "the default probability threshold of 0.50.\n"
    )

    report.write(
        "Threshold analysis is descriptive and "
        "does not constitute clinical threshold "
        "selection.\n"
    )

    report.write(
        "This is a research baseline, not a "
        "clinical diagnostic system.\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

print(
    "FILES CREATED:"
)

print(
    threshold_file
)

print(
    probability_file
)

print(
    false_negative_file
)

print(
    report_file
)

print()
print(
    "VOICE ERROR ANALYSIS: COMPLETE"
)

print("=" * 70)