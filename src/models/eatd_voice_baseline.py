from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# EATD VOICE BASELINE MODEL
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TARGET CONFIGURATION
# ============================================================

# EATD standardized SDS threshold:
# new_label > 52 -> depressed
SDS_THRESHOLD = 52.0


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("EATD VOICE BASELINE MODEL")
print("=" * 70)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Participants: {df['participant'].nunique()}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "split",
    "participant",
    "new_label",
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# TARGET CREATION
# ============================================================

df["new_label"] = pd.to_numeric(
    df["new_label"],
    errors="coerce",
)

if df["new_label"].isna().any():
    raise ValueError(
        "Invalid or missing new_label values found."
    )


df["target"] = (
    df["new_label"] > SDS_THRESHOLD
).astype(int)


# ============================================================
# OFFICIAL TRAIN / VALIDATION SPLIT
# ============================================================

train_df = df[
    df["split"].astype(str).str.lower() == "train"
].copy()

validation_df = df[
    df["split"].astype(str).str.lower()
    == "validation"
].copy()


if train_df.empty:
    raise ValueError(
        "Training split is empty."
    )

if validation_df.empty:
    raise ValueError(
        "Validation split is empty."
    )


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


if not feature_columns:
    raise ValueError(
        "No acoustic feature columns found."
    )


# ============================================================
# FEATURE MATRICES
# ============================================================

X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    "target"
].copy()

X_validation = validation_df[
    feature_columns
].copy()

y_validation = validation_df[
    "target"
].copy()


# ============================================================
# NUMERIC VALIDATION
# ============================================================

for column in feature_columns:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce",
    )

    X_validation[column] = pd.to_numeric(
        X_validation[column],
        errors="coerce",
    )


if np.isinf(
    X_train.to_numpy()
).any():
    raise ValueError(
        "Infinite values found in training features."
    )


if np.isinf(
    X_validation.to_numpy()
).any():
    raise ValueError(
        "Infinite values found in validation features."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("=" * 70)
print("OFFICIAL DATA SPLIT")
print("=" * 70)

print(
    f"Training participants: {len(train_df)}"
)

print(
    f"Validation participants: {len(validation_df)}"
)

print()
print("Training target distribution:")

print(
    y_train.value_counts()
    .sort_index()
    .rename(
        index={
            0: "non-depressed",
            1: "depressed",
        }
    )
)

print()
print("Validation target distribution:")

print(
    y_validation.value_counts()
    .sort_index()
    .rename(
        index={
            0: "non-depressed",
            1: "depressed",
        }
    )
)

print()
print(
    f"Number of acoustic features: "
    f"{len(feature_columns)}"
)

print(
    f"Training matrix: {X_train.shape}"
)

print(
    f"Validation matrix: {X_validation.shape}"
)


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name,
):

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

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

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "model_object": model,
        "predictions": predictions,
        "probabilities": probabilities,
    }


# ============================================================
# TRAIN + VALIDATE
# ============================================================

print()
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


logistic_result = evaluate_model(
    logistic_model,
    "Logistic Regression",
)


random_forest_result = evaluate_model(
    random_forest_model,
    "Random Forest",
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            key: value
            for key, value in logistic_result.items()
            if key not in [
                "model_object",
                "predictions",
                "probabilities",
            ]
        },
        {
            key: value
            for key, value in random_forest_result.items()
            if key not in [
                "model_object",
                "predictions",
                "probabilities",
            ]
        },
    ]
)


print()
print(
    results.to_string(
        index=False
    )
)


# ============================================================
# DETAILED CLASSIFICATION REPORT
# ============================================================

for result in [
    logistic_result,
    random_forest_result,
]:

    print()
    print("-" * 70)

    print(
        result["model"]
    )

    print("-" * 70)

    print(
        classification_report(
            y_validation,
            result["predictions"],
            target_names=[
                "non-depressed",
                "depressed",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_validation,
            result["predictions"],
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    REPORT_DIR
    / "eatd_voice_baseline_results.csv"
)

results.to_csv(
    results_file,
    index=False,
)


# ============================================================
# SAVE MODELS
# ============================================================

logistic_model_file = (
    MODEL_DIR
    / "logistic_regression.joblib"
)

random_forest_model_file = (
    MODEL_DIR
    / "random_forest.joblib"
)


joblib.dump(
    logistic_result["model_object"],
    logistic_model_file,
)

joblib.dump(
    random_forest_result["model_object"],
    random_forest_model_file,
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_estimator = (
    random_forest_result["model_object"]
    .named_steps["model"]
)

feature_importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance":
            rf_estimator.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


importance_file = (
    REPORT_DIR
    / "eatd_voice_feature_importance.csv"
)

feature_importance.to_csv(
    importance_file,
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

confusion_rows = []

for result in [
    logistic_result,
    random_forest_result,
]:

    matrix = confusion_matrix(
        y_validation,
        result["predictions"],
    )

    confusion_rows.append(
        {
            "model": result["model"],
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        }
    )


confusion_file = (
    REPORT_DIR
    / "eatd_voice_confusion_matrices.csv"
)

pd.DataFrame(
    confusion_rows
).to_csv(
    confusion_file,
    index=False,
)


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    REPORT_DIR
    / "eatd_voice_baseline_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "EATD Voice Baseline Model\n"
    )

    report.write(
        "=========================\n\n"
    )

    report.write(
        "Dataset: EATD-Corpus\n"
    )

    report.write(
        "Modality: Voice / Acoustic Features\n"
    )

    report.write(
        f"Acoustic features: "
        f"{len(feature_columns)}\n"
    )

    report.write(
        "Target: standardized SDS > 52\n"
    )

    report.write(
        f"Training participants: "
        f"{len(train_df)}\n"
    )

    report.write(
        f"Validation participants: "
        f"{len(validation_df)}\n\n"
    )

    report.write(
        "Validation Results\n"
    )

    report.write(
        "------------------\n"
    )

    report.write(
        results.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "Important limitations:\n"
    )

    report.write(
        "1. EATD is a relatively small dataset.\n"
    )

    report.write(
        "2. The target classes are imbalanced.\n"
    )

    report.write(
        "3. The official validation split is used "
        "as the held-out evaluation set.\n"
    )

    report.write(
        "4. Validation data was not used for model "
        "training.\n"
    )

    report.write(
        "5. The model is a research baseline and "
        "is not a clinical diagnostic system.\n"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)

print(
    "RESULTS SAVED:"
)

print(
    results_file
)

print()
print(
    "MODELS SAVED:"
)

print(
    logistic_model_file
)

print(
    random_forest_model_file
)

print()
print(
    "FEATURE IMPORTANCE:"
)

print(
    importance_file
)

print()
print(
    "CONFUSION MATRICES:"
)

print(
    confusion_file
)

print()
print(
    "REPORT:"
)

print(
    report_file
)

print()
print(
    "BASELINE MODEL VALIDATION: PASSED"
)

print("=" * 70)