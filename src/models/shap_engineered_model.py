from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

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
    print("SHAP ENGINEERED FEATURE EXPLAINABILITY")
    print("=" * 75)

    print(
        f"Training records : {len(df)}"
    )

    print(
        f"Engineered features : {X_engineered.shape[1]}"
    )

    # ---------------------------------------------------------
    # Scale features
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

    print(
        "[OK] Logistic Regression trained."
    )

    # ---------------------------------------------------------
    # Create SHAP background/sample
    # ---------------------------------------------------------

    sample_size = min(
        500,
        len(X_scaled),
    )

    sample_indices = (
        pd.Series(range(len(X_scaled)))
        .sample(
            n=sample_size,
            random_state=42,
        )
        .sort_values()
        .to_numpy()
    )

    X_sample = X_scaled[
        sample_indices
    ]

    print(
        f"SHAP sample size : {len(X_sample)}"
    )

    # ---------------------------------------------------------
    # SHAP Linear Explainer
    # ---------------------------------------------------------

    feature_names = list(
        X_engineered.columns
    )

    X_sample_df = pd.DataFrame(
        X_sample,
        columns=feature_names,
    )

    explainer = shap.LinearExplainer(
        model,
        X_sample_df,
    )

    shap_values = explainer(
        X_sample_df
    )

    print(
        "[OK] SHAP values calculated."
    )

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------

    mean_abs_shap = (
        abs(shap_values.values)
        .mean(axis=0)
    )

    mean_shap = (
        shap_values.values
        .mean(axis=0)
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "feature_group": [
                classify_feature_group(
                    feature
                )
                for feature in feature_names
            ],
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # Save complete SHAP importance table
    # ---------------------------------------------------------

    csv_path = (
        report_dir
        / "shap_engineered_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save top 30
    # ---------------------------------------------------------

    top_30 = importance_df.head(30)

    top_30_path = (
        report_dir
        / "shap_top_30_engineered_features.csv"
    )

    top_30.to_csv(
        top_30_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Group-level SHAP summary
    # ---------------------------------------------------------

    group_summary = (
        importance_df
        .groupby(
            "feature_group"
        )
        .agg(
            feature_count=(
                "feature",
                "count",
            ),
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),
            total_abs_shap=(
                "mean_abs_shap",
                "sum",
            ),
        )
        .sort_values(
            "total_abs_shap",
            ascending=False,
        )
        .reset_index()
    )

    group_path = (
        report_dir
        / "shap_feature_group_summary.csv"
    )

    group_summary.to_csv(
        group_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print top features
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP 30 FEATURES BY MEAN ABSOLUTE SHAP VALUE")
    print("=" * 75)

    print(
        top_30.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # Print group summary
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("SHAP FEATURE GROUP SUMMARY")
    print("=" * 75)

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ---------------------------------------------------------
    # SHAP bar plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    plt.tight_layout()

    plot_path = (
        report_dir
        / "shap_engineered_global_importance.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(csv_path)
    print(top_30_path)
    print(group_path)
    print(plot_path)

    print()
    print("=" * 75)
    print("SHAP ENGINEERED FEATURE ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()