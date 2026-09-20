from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "behavioral_phq9_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "studentlife"


def load_dataset():
    df = pd.read_csv(INPUT_PATH)

    required = {"uid", "type", "phq9_score"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def prepare_features(df):
    # Metadata columns must not become model features.
    excluded = {
        "uid",
        "type",
        "phq9_score",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    X = df[feature_columns].copy()
    y = df["phq9_score"].astype(float)
    groups = df["uid"].astype(str)

    # Ensure all model features are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, groups, feature_columns


def build_models():

    return {
        "dummy_mean": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    DummyRegressor(strategy="mean"),
                ),
            ]
        ),

        "ridge": Pipeline(
            [
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
                    Ridge(alpha=10.0),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate():

    df = load_dataset()

    X, y, groups, feature_columns = prepare_features(df)

    print("========================================")
    print("StudentLife Behavioral Baseline")
    print("========================================")
    print(f"Rows: {len(df)}")
    print(f"Participants: {groups.nunique()}")
    print(f"Features: {len(feature_columns)}")
    print(f"PHQ-9 range: {y.min():.0f} - {y.max():.0f}")
    print()

    models = build_models()

    seeds = [42, 7, 21, 100, 123]

    results = []

    for seed in seeds:

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                X,
                y,
                groups=groups,
            )
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = groups.iloc[train_idx]
        test_groups = groups.iloc[test_idx]

        # Explicit leakage check.
        overlap = set(train_groups) & set(test_groups)

        if overlap:
            raise AssertionError(
                f"Participant leakage detected: {overlap}"
            )

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            results.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_participants": len(
                        set(train_groups)
                    ),
                    "test_participants": len(
                        set(test_groups)
                    ),
                    "test_rows": len(test_idx),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )

    # Save detailed and summary results.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIR
        / "behavioral_baseline_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "behavioral_baseline_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("Participant leakage check: PASSED")
    print()
    print("Results across 5 participant-level splits:")
    print(summary.to_string(index=False))
    print()
    print(f"Detailed results: {results_path}")
    print(f"Summary: {summary_path}")
    print()
    print("Validation: PASSED")
    print("========================================")


if __name__ == "__main__":
    evaluate()