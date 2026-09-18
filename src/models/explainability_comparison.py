from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]
    report_dir = base_dir / "reports"

    coefficient_path = (
        report_dir
        / "engineered_feature_importance.csv"
    )

    shap_path = (
        report_dir
        / "shap_engineered_feature_importance.csv"
    )

    # ---------------------------------------------------------
    # Load existing explainability results
    # ---------------------------------------------------------

    coefficient_df = pd.read_csv(
        coefficient_path
    )

    shap_df = pd.read_csv(
        shap_path
    )

    print("=" * 75)
    print("EXPLAINABILITY METHOD COMPARISON")
    print("=" * 75)

    print(
        f"Coefficient features : {len(coefficient_df)}"
    )

    print(
        f"SHAP features         : {len(shap_df)}"
    )

    # ---------------------------------------------------------
    # Prepare coefficient importance
    # ---------------------------------------------------------

    coefficient_df = coefficient_df[
        [
            "feature",
            "coefficient",
            "absolute_coefficient",
        ]
    ].copy()

    coefficient_df = coefficient_df.rename(
        columns={
            "absolute_coefficient":
                "coefficient_importance"
        }
    )

    # ---------------------------------------------------------
    # Prepare SHAP importance
    # ---------------------------------------------------------

    shap_df = shap_df[
        [
            "feature",
            "mean_abs_shap",
        ]
    ].copy()

    shap_df = shap_df.rename(
        columns={
            "mean_abs_shap":
                "shap_importance"
        }
    )

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    merged = coefficient_df.merge(
        shap_df,
        on="feature",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 109:
        raise ValueError(
            f"Expected 109 matched features, "
            f"found {len(merged)}."
        )

    print(
        f"Matched features      : {len(merged)}"
    )

    # ---------------------------------------------------------
    # Rank features
    # ---------------------------------------------------------

    merged["coefficient_rank"] = (
        merged[
            "coefficient_importance"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    merged["shap_rank"] = (
        merged[
            "shap_importance"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Spearman rank correlation
    # ---------------------------------------------------------

    correlation, p_value = spearmanr(
        merged["coefficient_importance"],
        merged["shap_importance"],
    )

    # ---------------------------------------------------------
    # Top-N overlap
    # ---------------------------------------------------------

    def top_n_overlap(n: int) -> tuple[int, float]:

        coefficient_top = set(
            merged
            .nlargest(
                n,
                "coefficient_importance",
            )["feature"]
        )

        shap_top = set(
            merged
            .nlargest(
                n,
                "shap_importance",
            )["feature"]
        )

        overlap = len(
            coefficient_top
            & shap_top
        )

        percentage = (
            overlap / n
        ) * 100

        return overlap, percentage

    top_10_overlap, top_10_percentage = (
        top_n_overlap(10)
    )

    top_20_overlap, top_20_percentage = (
        top_n_overlap(20)
    )

    # ---------------------------------------------------------
    # Rank difference
    # ---------------------------------------------------------

    merged["rank_difference"] = (
        merged["coefficient_rank"]
        - merged["shap_rank"]
    ).abs()

    mean_rank_difference = (
        merged["rank_difference"]
        .mean()
    )

    # ---------------------------------------------------------
    # Save detailed comparison
    # ---------------------------------------------------------

    comparison_path = (
        report_dir
        / "explainability_comparison.csv"
    )

    merged.sort_values(
        "shap_rank"
    ).to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "matched_features": len(merged),
                "spearman_correlation": correlation,
                "spearman_p_value": p_value,
                "top_10_overlap": top_10_overlap,
                "top_10_overlap_percent": top_10_percentage,
                "top_20_overlap": top_20_overlap,
                "top_20_overlap_percent": top_20_percentage,
                "mean_absolute_rank_difference":
                    mean_rank_difference,
            }
        ]
    )

    summary_path = (
        report_dir
        / "explainability_comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("RANK CORRELATION")
    print("=" * 75)

    print(
        f"Spearman correlation : "
        f"{correlation:.6f}"
    )

    print(
        f"p-value              : "
        f"{p_value:.6f}"
    )

    print()
    print("=" * 75)
    print("TOP-N FEATURE OVERLAP")
    print("=" * 75)

    print(
        f"Top-10 overlap : "
        f"{top_10_overlap}/10 "
        f"({top_10_percentage:.1f}%)"
    )

    print(
        f"Top-20 overlap : "
        f"{top_20_overlap}/20 "
        f"({top_20_percentage:.1f}%)"
    )

    print()
    print("=" * 75)
    print("RANK DIFFERENCE")
    print("=" * 75)

    print(
        f"Mean absolute rank difference : "
        f"{mean_rank_difference:.4f}"
    )

    # ---------------------------------------------------------
    # Show top features from both methods
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("TOP 10 BY COEFFICIENT IMPORTANCE")
    print("=" * 75)

    print(
        merged
        .nlargest(
            10,
            "coefficient_importance",
        )[
            [
                "feature",
                "coefficient_importance",
                "coefficient_rank",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 75)
    print("TOP 10 BY SHAP IMPORTANCE")
    print("=" * 75)

    print(
        merged
        .nlargest(
            10,
            "shap_importance",
        )[
            [
                "feature",
                "shap_importance",
                "shap_rank",
            ]
        ]
        .to_string(
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

    print(comparison_path)
    print(summary_path)

    print()
    print("=" * 75)
    print("EXPLAINABILITY COMPARISON COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()