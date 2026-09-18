from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import extract_engineered_features


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
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(dataset_path)

    y = df["label"]

    X_engineered = extract_engineered_features(df)

    print("=" * 75)
    print("SHAP INDIVIDUAL PREDICTION EXPLANATION")
    print("=" * 75)

    print(
        f"Training records : {len(df)}"
    )

    print(
        f"Engineered features : {X_engineered.shape[1]}"
    )

    # ---------------------------------------------------------
    # Scale features while preserving DataFrame names
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled_array = scaler.fit_transform(
        X_engineered
    )

    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X_engineered.columns,
        index=X_engineered.index,
    )

    # ---------------------------------------------------------
    # Train model
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
        "[OK] Model trained."
    )

    # ---------------------------------------------------------
    # Select deterministic sample
    # ---------------------------------------------------------

    sample_index = 42

    X_sample = X_scaled.iloc[
        [sample_index]
    ]

    actual_label = int(
        y.iloc[sample_index]
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    predicted_class = int(
        model.predict(
            X_sample
        )[0]
    )

    probability_class_1 = float(
        model.predict_proba(
            X_sample
        )[0, 1]
    )

    # ---------------------------------------------------------
    # SHAP Linear Explainer
    # ---------------------------------------------------------

    explainer = shap.LinearExplainer(
        model,
        X_scaled,
    )

    shap_explanation = explainer(
        X_sample
    )

    shap_values = (
        shap_explanation.values[0]
    )

    base_value = float(
        shap_explanation.base_values[0]
    )

    feature_names = list(
        X_engineered.columns
    )

    # ---------------------------------------------------------
    # Explanation dataframe
    # ---------------------------------------------------------

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value": X_engineered.iloc[
                sample_index
            ].values,
            "shap_value": shap_values,
        }
    )

    explanation_df[
        "absolute_shap_value"
    ] = explanation_df[
        "shap_value"
    ].abs()

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Non-zero contributors
    # ---------------------------------------------------------

    non_zero = explanation_df[
        explanation_df["absolute_shap_value"] > 1e-10
    ].copy()

    positive = (
        non_zero[
            non_zero["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .head(10)
    )

    negative = (
        non_zero[
            non_zero["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .head(10)
    )

    # ---------------------------------------------------------
    # Save complete explanation
    # ---------------------------------------------------------

    explanation_path = (
        report_dir
        / "shap_individual_explanation.csv"
    )

    explanation_df.to_csv(
        explanation_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save positive contributors
    # ---------------------------------------------------------

    positive_path = (
        report_dir
        / "shap_individual_positive_contributors.csv"
    )

    positive.to_csv(
        positive_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save negative contributors
    # ---------------------------------------------------------

    negative_path = (
        report_dir
        / "shap_individual_negative_contributors.csv"
    )

    negative.to_csv(
        negative_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print sample information
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("SAMPLE INFORMATION")
    print("=" * 75)

    print(
        f"Sample index             : {sample_index}"
    )

    print(
        f"Actual label             : {actual_label}"
    )

    print(
        f"Predicted class          : {predicted_class}"
    )

    print(
        f"Class-1 model score      : {probability_class_1:.6f}"
    )

    print(
        f"SHAP base value          : {base_value:.6f}"
    )

    print(
        f"Non-zero SHAP features   : {len(non_zero)}"
    )

    # ---------------------------------------------------------
    # Positive contributors
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP POSITIVE SHAP CONTRIBUTORS")
    print("=" * 75)

    if positive.empty:
        print("No positive contributors found.")
    else:
        print(
            positive.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # ---------------------------------------------------------
    # Negative contributors
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP NEGATIVE SHAP CONTRIBUTORS")
    print("=" * 75)

    if negative.empty:
        print("No negative contributors found.")
    else:
        print(
            negative.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # ---------------------------------------------------------
    # Verify SHAP additive relationship
    # ---------------------------------------------------------

    shap_sum = (
        base_value
        + shap_values.sum()
    )

    model_decision = float(
        model.decision_function(
            X_sample
        )[0]
    )

    print()
    print("=" * 75)
    print("SHAP ADDITIVITY CHECK")
    print("=" * 75)

    print(
        f"Base + SHAP sum        : {shap_sum:.8f}"
    )

    print(
        f"Model decision value   : {model_decision:.8f}"
    )

    print(
        f"Absolute difference    : "
        f"{abs(shap_sum - model_decision):.8f}"
    )

    # ---------------------------------------------------------
    # Waterfall plot
    # ---------------------------------------------------------

    plot_path = (
        report_dir
        / "shap_individual_waterfall.png"
    )

    shap.plots.waterfall(
        shap_explanation[0],
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = pd.DataFrame(
        [
            {
                "sample_index": sample_index,
                "actual_label": actual_label,
                "predicted_class": predicted_class,
                "probability_class_1": probability_class_1,
                "shap_base_value": base_value,
                "non_zero_shap_features": len(non_zero),
                "shap_sum": shap_sum,
                "model_decision_value": model_decision,
                "additivity_absolute_difference": abs(
                    shap_sum - model_decision
                ),
            }
        ]
    )

    metadata_path = (
        report_dir
        / "shap_individual_metadata.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(explanation_path)
    print(positive_path)
    print(negative_path)
    print(plot_path)
    print(metadata_path)

    print()
    print("=" * 75)
    print("INDIVIDUAL SHAP EXPLANATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()