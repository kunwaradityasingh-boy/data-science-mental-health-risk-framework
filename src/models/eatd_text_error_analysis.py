from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
    / "eatd_text_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "eatd_text"
    / "tfidf_logistic_regression.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_REPORT = REPORT_DIR / "eatd_text_threshold_analysis.csv"
PROBABILITY_REPORT = REPORT_DIR / "eatd_text_probability_summary.csv"
FALSE_NEGATIVE_REPORT = REPORT_DIR / "eatd_text_false_negatives.csv"
FINAL_REPORT = REPORT_DIR / "eatd_text_error_analysis_report.txt"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target"

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

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "split",
    "participant",
    "label",
    "new_label",
    "text",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# ============================================================
# TARGET
# ============================================================

# EATD classification target:
# new_label > 52 => depressed
# new_label <= 52 => non-depressed

df[TARGET_COLUMN] = (
    pd.to_numeric(df["new_label"], errors="coerce") > 52
).astype(int)


# ============================================================
# PRESERVE OFFICIAL EATD SPLIT
# ============================================================

train_df = df[df["split"].astype(str).str.lower() == "train"].copy()

validation_df = df[
    df["split"].astype(str).str.lower() == "validation"
].copy()

if train_df.empty:
    raise ValueError("Training split is empty.")

if validation_df.empty:
    raise ValueError("Validation split is empty.")


# ============================================================
# RESET INDEX
# IMPORTANT:
# Keeps probability arrays aligned with validation rows.
# ============================================================

validation_df = validation_df.reset_index(drop=True)


# ============================================================
# LOAD TRAINED PIPELINE
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

X_validation = validation_df["text"].fillna("").astype(str)

y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()

probabilities = model.predict_proba(X_validation)[:, 1]

default_predictions = (probabilities >= 0.50).astype(int)


# ============================================================
# GLOBAL METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_validation,
    probabilities,
)

pr_auc = average_precision_score(
    y_validation,
    probabilities,
)

default_accuracy = accuracy_score(
    y_validation,
    default_predictions,
)

default_precision = precision_score(
    y_validation,
    default_predictions,
    zero_division=0,
)

default_recall = recall_score(
    y_validation,
    default_predictions,
    zero_division=0,
)

default_f1 = f1_score(
    y_validation,
    default_predictions,
    zero_division=0,
)

default_cm = confusion_matrix(
    y_validation,
    default_predictions,
    labels=[0, 1],
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

threshold_rows = []

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    threshold_rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(
                y_validation,
                predictions,
            ),
            "precision": precision_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_validation,
                predictions,
                zero_division=0,
            ),
            "predicted_positive": int(
                predictions.sum()
            ),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

threshold_df.to_csv(
    THRESHOLD_REPORT,
    index=False,
)


# ============================================================
# PROBABILITY SUMMARY
# ============================================================

probability_summary = pd.DataFrame(
    [
        {
            "model": "TF-IDF + Logistic Regression",
            "dataset": "EATD validation",
            "validation_samples": len(validation_df),
            "positive_samples": int(y_validation.sum()),
            "negative_samples": int(
                (y_validation == 0).sum()
            ),
            "probability_min": float(
                probabilities.min()
            ),
            "probability_mean": float(
                probabilities.mean()
            ),
            "probability_median": float(
                np.median(probabilities)
            ),
            "probability_std": float(
                probabilities.std()
            ),
            "probability_max": float(
                probabilities.max()
            ),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }
    ]
)

probability_summary.to_csv(
    PROBABILITY_REPORT,
    index=False,
)


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

false_negative_mask = (
    (y_validation == 1)
    & (default_predictions == 0)
)

false_negative_df = validation_df.loc[
    false_negative_mask
].copy()

false_negative_df["predicted_probability"] = probabilities[
    false_negative_mask
]

false_negative_df["predicted_class"] = default_predictions[
    false_negative_mask
]

false_negative_df["error_type"] = "false_negative"

# Do not export the full text by default.
# Export only useful metadata to keep the report compact.
false_negative_columns = [
    "split",
    "participant",
    "label",
    "new_label",
    "predicted_probability",
    "predicted_class",
    "error_type",
]

false_negative_df[
    false_negative_columns
].sort_values(
    "predicted_probability"
).to_csv(
    FALSE_NEGATIVE_REPORT,
    index=False,
)


# ============================================================
# CLASS-CONDITIONED PROBABILITY SUMMARY
# ============================================================

non_depressed_probs = probabilities[
    y_validation == 0
]

depressed_probs = probabilities[
    y_validation == 1
]

class_probability_rows = [
    {
        "class": 0,
        "class_name": "non_depressed",
        "samples": len(non_depressed_probs),
        "mean_probability": float(
            non_depressed_probs.mean()
        ),
        "median_probability": float(
            np.median(non_depressed_probs)
        ),
        "max_probability": float(
            non_depressed_probs.max()
        ),
    },
    {
        "class": 1,
        "class_name": "depressed",
        "samples": len(depressed_probs),
        "mean_probability": float(
            depressed_probs.mean()
        ),
        "median_probability": float(
            np.median(depressed_probs)
        ),
        "max_probability": float(
            depressed_probs.max()
        ),
    },
]

class_probability_df = pd.DataFrame(
    class_probability_rows
)

class_probability_path = (
    REPORT_DIR
    / "eatd_text_class_probability_summary.csv"
)

class_probability_df.to_csv(
    class_probability_path,
    index=False,
)


# ============================================================
# ERROR COUNTS
# ============================================================

tn, fp, fn, tp = default_cm.ravel()


# ============================================================
# FINAL REPORT
# ============================================================

report_lines = []

report_lines.append(
    "EATD TEXT ERROR AND THRESHOLD ANALYSIS REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append("")

report_lines.append(
    "Dataset: EATD-Corpus"
)

report_lines.append(
    "Modality: Text"
)

report_lines.append(
    "Model: Character-level TF-IDF + Logistic Regression"
)

report_lines.append(
    "Target: new_label > 52"
)

report_lines.append(
    "Official split: train / validation"
)

report_lines.append("")

report_lines.append(
    "VALIDATION DATASET"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    f"Validation samples: {len(validation_df)}"
)

report_lines.append(
    f"Non-depressed: {(y_validation == 0).sum()}"
)

report_lines.append(
    f"Depressed: {(y_validation == 1).sum()}"
)

report_lines.append("")

report_lines.append(
    "DEFAULT THRESHOLD = 0.50"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    f"Accuracy: {default_accuracy:.6f}"
)

report_lines.append(
    f"Precision: {default_precision:.6f}"
)

report_lines.append(
    f"Recall: {default_recall:.6f}"
)

report_lines.append(
    f"F1: {default_f1:.6f}"
)

report_lines.append(
    f"ROC-AUC: {roc_auc:.6f}"
)

report_lines.append(
    f"PR-AUC: {pr_auc:.6f}"
)

report_lines.append("")

report_lines.append(
    "Confusion Matrix [TN FP; FN TP]"
)

report_lines.append(
    f"[[{tn}, {fp}], [{fn}, {tp}]]"
)

report_lines.append("")

report_lines.append(
    "PROBABILITY DISTRIBUTION"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    f"Minimum: {probabilities.min():.8f}"
)

report_lines.append(
    f"Mean: {probabilities.mean():.8f}"
)

report_lines.append(
    f"Median: {np.median(probabilities):.8f}"
)

report_lines.append(
    f"Maximum: {probabilities.max():.8f}"
)

report_lines.append("")

report_lines.append(
    "FALSE NEGATIVE ANALYSIS"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    f"False negatives at threshold 0.50: {fn}"
)

report_lines.append(
    f"True positives at threshold 0.50: {tp}"
)

report_lines.append("")

report_lines.append(
    "THRESHOLD ANALYSIS"
)

report_lines.append(
    "-" * 60
)

for row in threshold_rows:

    report_lines.append(
        "Threshold "
        f"{row['threshold']:.2f}: "
        f"Accuracy={row['accuracy']:.6f}, "
        f"Precision={row['precision']:.6f}, "
        f"Recall={row['recall']:.6f}, "
        f"F1={row['f1']:.6f}, "
        f"PredictedPositive={row['predicted_positive']}"
    )

report_lines.append("")

report_lines.append(
    "INTERPRETATION"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    "The validation results are evaluated on the official "
    "EATD validation split."
)

report_lines.append(
    "Accuracy must be interpreted together with precision, "
    "recall, F1, ROC-AUC and PR-AUC because the validation "
    "set contains substantially more non-depressed than "
    "depressed participants."
)

report_lines.append(
    "Threshold analysis is descriptive and does not "
    "constitute clinical threshold selection."
)

report_lines.append(
    "The model is a research baseline and must not be "
    "interpreted as a diagnostic system."
)

report_lines.append(
    "The text model currently shows weak discrimination "
    "for the depressed class on the held-out validation set."
)

report_lines.append(
    "Changing the classification threshold cannot by itself "
    "establish that the underlying text representation is "
    "clinically or scientifically adequate."
)

report_lines.append("")

report_lines.append(
    "PHASE 7 ERROR ANALYSIS STATUS"
)

report_lines.append(
    "-" * 60
)

report_lines.append(
    "Probability analysis: COMPLETE"
)

report_lines.append(
    "Threshold analysis: COMPLETE"
)

report_lines.append(
    "False-negative analysis: COMPLETE"
)

report_lines.append(
    "Validation metrics: COMPLETE"
)

report_lines.append(
    "Research limitation documentation: COMPLETE"
)

report_lines.append("")

report_lines.append(
    "Phase 7 text error-analysis component: COMPLETE"
)


FINAL_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# FINAL VALIDATION CHECKS
# ============================================================

expected_outputs = [
    THRESHOLD_REPORT,
    PROBABILITY_REPORT,
    FALSE_NEGATIVE_REPORT,
    class_probability_path,
    FINAL_REPORT,
]

missing_outputs = [
    str(path)
    for path in expected_outputs
    if not path.exists()
]

if missing_outputs:
    raise RuntimeError(
        "Expected report files were not created:\n"
        + "\n".join(missing_outputs)
    )


print()
print("=" * 70)
print("EATD TEXT ERROR ANALYSIS")
print("=" * 70)

print(f"Validation samples: {len(validation_df)}")
print(f"Depressed samples: {int(y_validation.sum())}")
print(
    f"Non-depressed samples: "
    f"{int((y_validation == 0).sum())}"
)

print()
print("DEFAULT THRESHOLD = 0.50")
print(f"Accuracy : {default_accuracy:.6f}")
print(f"Precision: {default_precision:.6f}")
print(f"Recall   : {default_recall:.6f}")
print(f"F1       : {default_f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")
print(f"PR-AUC   : {pr_auc:.6f}")

print()
print("CONFUSION MATRIX")
print(default_cm)

print()
print("PROBABILITY DISTRIBUTION")
print(f"Min    : {probabilities.min():.8f}")
print(f"Mean   : {probabilities.mean():.8f}")
print(f"Median : {np.median(probabilities):.8f}")
print(f"Max    : {probabilities.max():.8f}")

print()
print("FALSE NEGATIVES")
print(f"Count: {fn}")

print()
print("OUTPUT FILES")
for path in expected_outputs:
    print(f"[OK] {path}")

print()
print("EATD TEXT ERROR ANALYSIS: PASSED")