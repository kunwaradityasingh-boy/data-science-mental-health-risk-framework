from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
# EATD TEXT BASELINE MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
    / "eatd_text_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "eatd_text"
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
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE,
    encoding="utf-8-sig",
)

print("=" * 70)
print("EATD TEXT BASELINE MODEL")
print("=" * 70)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "split",
    "participant",
    "text",
    "target",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# OFFICIAL SPLIT
# ============================================================

train_df = df[
    df["split"].astype(str).str.lower()
    == "train"
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
# TEXT VALIDATION
# ============================================================

train_df["text"] = (
    train_df["text"]
    .fillna("")
    .astype(str)
)

validation_df["text"] = (
    validation_df["text"]
    .fillna("")
    .astype(str)
)


if (
    train_df["text"].str.strip() == ""
).any():

    raise ValueError(
        "Empty training text detected."
    )


if (
    validation_df["text"].str.strip() == ""
).any():

    raise ValueError(
        "Empty validation text detected."
    )


# ============================================================
# TARGET
# ============================================================

y_train = (
    train_df["target"]
    .astype(int)
)

y_validation = (
    validation_df["target"]
    .astype(int)
)


X_train = train_df["text"]

X_validation = validation_df["text"]


# ============================================================
# TARGET VALIDATION
# ============================================================

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains only one target class."
    )

if y_validation.nunique() < 2:
    raise ValueError(
        "Validation data contains only one target class."
    )


# ============================================================
# INFORMATION
# ============================================================

print()
print("=" * 70)
print("OFFICIAL DATA SPLIT")
print("=" * 70)

print(
    f"Training participants: "
    f"{len(train_df)}"
)

print(
    f"Validation participants: "
    f"{len(validation_df)}"
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


# ============================================================
# TEXT MODEL
# ============================================================

# Character-level TF-IDF is used because EATD text
# is Chinese and the dataset is small.
#
# The vectorizer is fitted ONLY on training text.
# Validation text is transformed using the fitted
# training vocabulary.

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=20000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# TF-IDF INFORMATION
# ============================================================

vectorizer = (
    model
    .named_steps["tfidf"]
)

print(
    f"TF-IDF vocabulary size: "
    f"{len(vectorizer.vocabulary_)}"
)


# ============================================================
# VALIDATION PREDICTION
# ============================================================

predictions = model.predict(
    X_validation
)

probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "Character TF-IDF + Logistic Regression",
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
        }
    ]
)


print()
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_validation,
        predictions,
        target_names=[
            "non-depressed",
            "depressed",
        ],
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion matrix:")

print(
    matrix
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR
    / "tfidf_logistic_regression.joblib"
)

joblib.dump(
    model,
    model_file,
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    REPORT_DIR
    / "eatd_text_baseline_results.csv"
)

results.to_csv(
    results_file,
    index=False,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "split": validation_df[
            "split"
        ].values,

        "participant": validation_df[
            "participant"
        ].values,

        "true_target": y_validation.values,

        "predicted_target":
            predictions,

        "predicted_probability":
            probabilities,

        "new_label": validation_df[
            "new_label"
        ].values,
    }
)


prediction_file = (
    REPORT_DIR
    / "eatd_text_validation_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_file = (
    REPORT_DIR
    / "eatd_text_confusion_matrix.csv"
)

pd.DataFrame(
    [
        {
            "model":
                "Character TF-IDF + Logistic Regression",

            "true_negative":
                int(matrix[0, 0]),

            "false_positive":
                int(matrix[0, 1]),

            "false_negative":
                int(matrix[1, 0]),

            "true_positive":
                int(matrix[1, 1]),
        }
    ]
).to_csv(
    confusion_file,
    index=False,
)


# ============================================================
# TOP TF-IDF / MODEL FEATURES
# ============================================================

classifier = (
    model
    .named_steps["classifier"]
)

feature_names = (
    vectorizer.get_feature_names_out()
)

coefficients = (
    classifier.coef_[0]
)

feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient":
            np.abs(coefficients),
    }
).sort_values(
    "absolute_coefficient",
    ascending=False,
)


feature_file = (
    REPORT_DIR
    / "eatd_text_feature_coefficients.csv"
)

feature_importance.to_csv(
    feature_file,
    index=False,
)


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    REPORT_DIR
    / "eatd_text_baseline_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "EATD Text Baseline Model\n"
    )

    report.write(
        "========================\n\n"
    )

    report.write(
        "Dataset: EATD-Corpus\n"
    )

    report.write(
        "Modality: Text\n"
    )

    report.write(
        "Representation: Character-level TF-IDF\n"
    )

    report.write(
        "Classifier: Logistic Regression\n"
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
        f"{len(validation_df)}\n"
    )

    report.write(
        f"TF-IDF vocabulary size: "
        f"{len(vectorizer.vocabulary_)}\n\n"
    )

    report.write(
        "Validation metrics:\n"
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
        "3. The official validation split is "
        "used as the held-out evaluation set.\n"
    )

    report.write(
        "4. The TF-IDF vocabulary is fitted only "
        "on the training split.\n"
    )

    report.write(
        "5. This is a research baseline and not "
        "a clinical diagnostic system.\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

print("FILES SAVED:")

print(
    model_file
)

print(
    results_file
)

print(
    prediction_file
)

print(
    confusion_file
)

print(
    feature_file
)

print(
    report_file
)

print()
print(
    "EATD TEXT BASELINE: PASSED"
)

print("=" * 70)