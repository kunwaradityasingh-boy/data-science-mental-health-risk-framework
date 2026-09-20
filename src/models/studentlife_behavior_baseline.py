from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "behavior_fused_pre.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "studentlife_behavior"
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
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("=" * 65)
print("STUDENTLIFE BEHAVIORAL BASELINE MODEL")
print("=" * 65)

print(f"Dataset shape: {df.shape}")
print(f"Unique participants: {df['uid'].nunique()}")


# ============================================================
# TARGET
# ============================================================

TARGET = "phq9_score"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = [
    "uid",
    "phq9_score",
    "target_source",
    "target_definition",
    "dataset_role",
    "temporal_alignment",
]

feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]


if not feature_columns:
    raise ValueError(
        "No behavioral feature columns available."
    )


# ============================================================
# NUMERIC FEATURE CHECK
# ============================================================

X = df[feature_columns].copy()
y = df[TARGET].copy()

non_numeric = [
    column
    for column in X.columns
    if not pd.api.types.is_numeric_dtype(
        X[column]
    )
]

if non_numeric:
    raise ValueError(
        "Non-numeric feature columns found: "
        f"{non_numeric}"
    )


# Convert explicitly to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce",
)

y = pd.to_numeric(
    y,
    errors="coerce",
)


# ============================================================
# DATA VALIDATION
# ============================================================

if X.isna().any().any():
    raise ValueError(
        "Feature matrix contains missing values."
    )

if y.isna().any():
    raise ValueError(
        "Target contains missing values."
    )

if np.isinf(X.to_numpy()).any():
    raise ValueError(
        "Feature matrix contains infinite values."
    )

if np.isinf(y.to_numpy()).any():
    raise ValueError(
        "Target contains infinite values."
    )


print(f"Behavioral features: {len(feature_columns)}")
print(f"Target: {TARGET}")
print(
    f"Target range: "
    f"{y.min():.2f} - {y.max():.2f}"
)
print(
    f"Target mean: {y.mean():.2f}"
)


# ============================================================
# PARTICIPANT-LEVEL CROSS VALIDATION
# ============================================================
#
# Each participant occurs exactly once in this fused dataset.
# Therefore KFold operates at participant level without
# splitting repeated observations from the same participant.
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# MODEL 1 — RIDGE REGRESSION
# ============================================================

ridge_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            Ridge(alpha=1.0),
        ),
    ]
)


# ============================================================
# MODEL 2 — RANDOM FOREST REGRESSOR
# ============================================================

rf_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# CROSS-VALIDATION METRICS
# ============================================================

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def evaluate_model(
    model,
    model_name,
):
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    mae = -results["test_mae"]
    rmse = -results["test_rmse"]
    r2 = results["test_r2"]

    summary = {
        "model": model_name,
        "MAE_mean": mae.mean(),
        "MAE_std": mae.std(),
        "RMSE_mean": rmse.mean(),
        "RMSE_std": rmse.std(),
        "R2_mean": r2.mean(),
        "R2_std": r2.std(),
    }

    return summary


print()
print("=" * 65)
print("5-FOLD PARTICIPANT-LEVEL CROSS-VALIDATION")
print("=" * 65)

ridge_result = evaluate_model(
    ridge_pipeline,
    "Ridge Regression",
)

rf_result = evaluate_model(
    rf_pipeline,
    "Random Forest",
)

results = pd.DataFrame(
    [
        ridge_result,
        rf_result,
    ]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

for _, row in results.iterrows():

    print(
        f"{row['model']}"
    )

    print(
        f"  MAE : "
        f"{row['MAE_mean']:.4f} "
        f"+/- {row['MAE_std']:.4f}"
    )

    print(
        f"  RMSE: "
        f"{row['RMSE_mean']:.4f} "
        f"+/- {row['RMSE_std']:.4f}"
    )

    print(
        f"  R²  : "
        f"{row['R2_mean']:.4f} "
        f"+/- {row['R2_std']:.4f}"
    )

    print()


# ============================================================
# FIT FINAL BASELINE MODELS ON FULL DATA
# ============================================================

ridge_pipeline.fit(
    X,
    y,
)

rf_pipeline.fit(
    X,
    y,
)


# ============================================================
# SAVE MODELS
# ============================================================

ridge_path = (
    MODEL_DIR
    / "ridge_baseline.joblib"
)

rf_path = (
    MODEL_DIR
    / "random_forest_baseline.joblib"
)

joblib.dump(
    ridge_pipeline,
    ridge_path,
)

joblib.dump(
    rf_pipeline,
    rf_path,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_model = rf_pipeline.named_steps[
    "model"
]

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": rf_model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


importance_path = (
    REPORT_DIR
    / "studentlife_behavior_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False,
)


# ============================================================
# RESULTS REPORT
# ============================================================

results_path = (
    REPORT_DIR
    / "studentlife_behavior_baseline_results.csv"
)

results.to_csv(
    results_path,
    index=False,
)


# ============================================================
# TEXT REPORT
# ============================================================

report_path = (
    REPORT_DIR
    / "studentlife_behavior_baseline_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "StudentLife Behavioral Baseline Model\n"
    )
    report.write(
        "=====================================\n\n"
    )

    report.write(
        "Dataset role:\n"
        "Retrospective participant-level "
        "behavioral association baseline.\n\n"
    )

    report.write(
        "Important limitation:\n"
        "The available PHQ-9 table contains pre/post "
        "assessment type but no exact assessment timestamp. "
        "Therefore this experiment is NOT presented as "
        "temporally aligned early prediction.\n\n"
    )

    report.write(
        f"Participants: {len(df)}\n"
    )

    report.write(
        f"Behavioral features: "
        f"{len(feature_columns)}\n"
    )

    report.write(
        f"Target: PHQ-9 pre-assessment score\n"
    )

    report.write(
        f"Target range: "
        f"{y.min():.2f} - {y.max():.2f}\n\n"
    )

    report.write(
        "5-Fold Participant-Level Cross-Validation\n"
    )
    report.write(
        "------------------------------------------\n"
    )

    for _, row in results.iterrows():

        report.write(
            f"\n{row['model']}\n"
        )

        report.write(
            f"MAE: "
            f"{row['MAE_mean']:.4f} "
            f"+/- {row['MAE_std']:.4f}\n"
        )

        report.write(
            f"RMSE: "
            f"{row['RMSE_mean']:.4f} "
            f"+/- {row['RMSE_std']:.4f}\n"
        )

        report.write(
            f"R2: "
            f"{row['R2_mean']:.4f} "
            f"+/- {row['R2_std']:.4f}\n"
        )


# ============================================================
# TOP FEATURES
# ============================================================

print("=" * 65)
print("TOP BEHAVIORAL FEATURES")
print("=" * 65)

print(
    importance.head(15).to_string(
        index=False
    )
)

print()
print("=" * 65)
print("FILES SAVED")
print("=" * 65)

print(
    f"Ridge model:\n{ridge_path}"
)

print(
    f"Random Forest model:\n{rf_path}"
)

print(
    f"Results:\n{results_path}"
)

print(
    f"Feature importance:\n{importance_path}"
)

print(
    f"Report:\n{report_path}"
)

print()
print("VALIDATION: PASSED")
print("=" * 65)