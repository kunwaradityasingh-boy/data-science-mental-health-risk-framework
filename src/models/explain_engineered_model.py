from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import (
    extract_engineered_features,
)
from src.features.feature_groups import (
    classify_feature_group,
)


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    dataset_path = (
        base_dir
        / "data"
        / "raw"
        / "dreaddit"
        / "extracted"
        / "dreaddit-train.csv"
    )

    report_dir = base_dir / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

    df = pd.read_csv(dataset_path)

    y = df["label"]

    X_engineered = extract_engineered_features(df)

    print("=" * 75)
    print("ENGINEERED FEATURE MODEL EXPLAINABILITY")
    print("=" * 75)

    print(
        f"Training records : {len(df)}"
    )

    print(
        f"Engineered features : {X_engineered.shape[1]}"
    )

    # ---------------------------------------------------------
    # Scale engineered features
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_engineered
    )

    # ---------------------------------------------------------
    # Train Logistic Regression
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_scaled,
        y,
    )

    # ---------------------------------------------------------
    # Extract coefficients
    # ---------------------------------------------------------

    feature_names = X_engineered.columns

    coefficients = model.coef_[0]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
            "feature_group": [
                classify_feature_group(
                    feature
                )
                for feature in feature_names
            ],
        }
    )

    # ---------------------------------------------------------
    # Sort by absolute importance
    # ---------------------------------------------------------

    absolute_importance = (
        importance_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .copy()
    )

    positive_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(30)
        .copy()
    )

    negative_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(30)
        .copy()
    )

    # ---------------------------------------------------------
    # Feature-group summary
    # ---------------------------------------------------------

    group_summary = (
        importance_df
        .groupby(
            "feature_group"
        )
        .agg(
            feature_count=("feature", "count"),
            mean_absolute_coefficient=(
                "absolute_coefficient",
                "mean",
            ),
            max_absolute_coefficient=(
                "absolute_coefficient",
                "max",
            ),
        )
        .sort_values(
            "mean_absolute_coefficient",
            ascending=False,
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Save reports
    # ---------------------------------------------------------

    absolute_path = (
        report_dir
        / "engineered_feature_importance.csv"
    )

    positive_path = (
        report_dir
        / "engineered_positive_features.csv"
    )

    negative_path = (
        report_dir
        / "engineered_negative_features.csv"
    )

    group_path = (
        report_dir
        / "engineered_feature_group_importance.csv"
    )

    absolute_importance.to_csv(
        absolute_path,
        index=False,
    )

    positive_features.to_csv(
        positive_path,
        index=False,
    )

    negative_features.to_csv(
        negative_path,
        index=False,
    )

    group_summary.to_csv(
        group_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display top absolute features
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP ENGINEERED FEATURES BY ABSOLUTE COEFFICIENT")
    print("=" * 75)

    print(
        absolute_importance.head(30).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Display positive features
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP FEATURES CONTRIBUTING TOWARD CLASS 1")
    print("=" * 75)

    print(
        positive_features.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Display negative features
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP FEATURES CONTRIBUTING TOWARD CLASS 0")
    print("=" * 75)

    print(
        negative_features.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Display group summary
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("FEATURE GROUP IMPORTANCE")
    print("=" * 75)

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(absolute_path)
    print(positive_path)
    print(negative_path)
    print(group_path)

    print()
    print("=" * 75)
    print("ENGINEERED FEATURE EXPLAINABILITY COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()