from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    prediction_path = (
        base_dir
        / "reports"
        / "final_holdout_predictions.csv"
    )

    report_dir = base_dir / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # LOAD PREDICTIONS
    # =========================================================

    df = pd.read_csv(prediction_path)

    print("=" * 75)
    print("DREAddit FINAL HOLDOUT ERROR ANALYSIS")
    print("=" * 75)

    print(f"Prediction rows : {len(df)}")
    print(f"Experiments     : {df['experiment'].nunique()}")

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    expected_experiments = {
        "text_only",
        "text_plus_social",
        "text_plus_all_engineered",
    }

    actual_experiments = set(
        df["experiment"].unique()
    )

    if actual_experiments != expected_experiments:
        raise ValueError(
            "Unexpected experiment names found."
        )

    required_columns = {
        "experiment",
        "id",
        "post_id",
        "subreddit",
        "actual_label",
        "predicted_label",
        "class_1_score",
        "text_length",
        "error_type",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # =========================================================
    # ERROR TYPE DISTRIBUTION
    # =========================================================

    error_counts = (
        df.groupby(
            [
                "experiment",
                "error_type",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    # Calculate percentage within each experiment.
    experiment_totals = (
        df.groupby("experiment")
        .size()
        .rename("experiment_total")
        .reset_index()
    )

    error_counts = error_counts.merge(
        experiment_totals,
        on="experiment",
        how="left",
    )

    error_counts["percentage"] = (
        error_counts["count"]
        / error_counts["experiment_total"]
    ) * 100

    error_counts = error_counts.drop(
        columns=["experiment_total"]
    )

    error_counts_path = (
        report_dir
        / "error_analysis_counts.csv"
    )

    error_counts.to_csv(
        error_counts_path,
        index=False,
    )

    # =========================================================
    # PRINT ERROR DISTRIBUTION
    # =========================================================

    print()
    print("=" * 75)
    print("ERROR TYPE DISTRIBUTION")
    print("=" * 75)

    print(
        error_counts.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # =========================================================
    # SUBREDDIT ANALYSIS
    # =========================================================

    subreddit_analysis = (
        df.groupby(
            [
                "experiment",
                "subreddit",
            ]
        )
        .agg(
            records=("id", "count"),
            errors=(
                "error_type",
                lambda x: (
                    (x == "false_positive")
                    | (x == "false_negative")
                ).sum(),
            ),
            false_positives=(
                "error_type",
                lambda x: (
                    x == "false_positive"
                ).sum(),
            ),
            false_negatives=(
                "error_type",
                lambda x: (
                    x == "false_negative"
                ).sum(),
            ),
        )
        .reset_index()
    )

    subreddit_analysis["correct"] = (
        subreddit_analysis["records"]
        - subreddit_analysis["errors"]
    )

    subreddit_analysis["error_rate"] = (
        subreddit_analysis["errors"]
        / subreddit_analysis["records"]
    ) * 100

    subreddit_path = (
        report_dir
        / "error_analysis_subreddit.csv"
    )

    subreddit_analysis.to_csv(
        subreddit_path,
        index=False,
    )

    # =========================================================
    # SCORE BAND ANALYSIS
    # =========================================================

    df["score_distance_from_threshold"] = (
        df["class_1_score"] - 0.5
    ).abs()

    df["score_band"] = pd.cut(
        df["class_1_score"],
        bins=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        labels=[
            "0.0-0.2",
            "0.2-0.4",
            "0.4-0.6",
            "0.6-0.8",
            "0.8-1.0",
        ],
        include_lowest=True,
    )

    score_analysis = (
        df.groupby(
            [
                "experiment",
                "score_band",
            ],
            observed=True,
        )
        .agg(
            records=("id", "count"),
            errors=(
                "error_type",
                lambda x: (
                    (x == "false_positive")
                    | (x == "false_negative")
                ).sum(),
            ),
            false_positives=(
                "error_type",
                lambda x: (
                    x == "false_positive"
                ).sum(),
            ),
            false_negatives=(
                "error_type",
                lambda x: (
                    x == "false_negative"
                ).sum(),
            ),
        )
        .reset_index()
    )

    score_analysis["error_rate"] = (
        score_analysis["errors"]
        / score_analysis["records"]
    ) * 100

    score_path = (
        report_dir
        / "error_analysis_score_bands.csv"
    )

    score_analysis.to_csv(
        score_path,
        index=False,
    )

    # =========================================================
    # TEXT LENGTH ANALYSIS
    # =========================================================

    df["text_length_band"] = pd.cut(
        df["text_length"],
        bins=[
            -1,
            50,
            100,
            250,
            500,
            1000,
            float("inf"),
        ],
        labels=[
            "0-50",
            "51-100",
            "101-250",
            "251-500",
            "501-1000",
            "1000+",
        ],
    )

    length_analysis = (
        df.groupby(
            [
                "experiment",
                "text_length_band",
            ],
            observed=True,
        )
        .agg(
            records=("id", "count"),
            errors=(
                "error_type",
                lambda x: (
                    (x == "false_positive")
                    | (x == "false_negative")
                ).sum(),
            ),
            false_positives=(
                "error_type",
                lambda x: (
                    x == "false_positive"
                ).sum(),
            ),
            false_negatives=(
                "error_type",
                lambda x: (
                    x == "false_negative"
                ).sum(),
            ),
        )
        .reset_index()
    )

    length_analysis["error_rate"] = (
        length_analysis["errors"]
        / length_analysis["records"]
    ) * 100

    length_path = (
        report_dir
        / "error_analysis_text_length.csv"
    )

    length_analysis.to_csv(
        length_path,
        index=False,
    )

    # =========================================================
    # FALSE NEGATIVE CASES
    # =========================================================

    false_negatives = df[
        df["error_type"]
        == "false_negative"
    ].copy()

    false_negatives = (
        false_negatives
        .sort_values(
            "class_1_score",
            ascending=False,
        )
    )

    fn_path = (
        report_dir
        / "false_negative_cases.csv"
    )

    false_negatives.to_csv(
        fn_path,
        index=False,
    )

    # =========================================================
    # FALSE POSITIVE CASES
    # =========================================================

    false_positives = df[
        df["error_type"]
        == "false_positive"
    ].copy()

    false_positives = (
        false_positives
        .sort_values(
            "class_1_score",
            ascending=True,
        )
    )

    fp_path = (
        report_dir
        / "false_positive_cases.csv"
    )

    false_positives.to_csv(
        fp_path,
        index=False,
    )

    # =========================================================
    # CORRECTED SUMMARY
    # =========================================================

    summary = (
        df.groupby("experiment")
        .agg(
            total_records=("id", "count"),
            true_positives=(
                "error_type",
                lambda x: (
                    x == "true_positive"
                ).sum(),
            ),
            true_negatives=(
                "error_type",
                lambda x: (
                    x == "true_negative"
                ).sum(),
            ),
            false_positives=(
                "error_type",
                lambda x: (
                    x == "false_positive"
                ).sum(),
            ),
            false_negatives=(
                "error_type",
                lambda x: (
                    x == "false_negative"
                ).sum(),
            ),
        )
        .reset_index()
    )

    summary["total_errors"] = (
        summary["false_positives"]
        + summary["false_negatives"]
    )

    summary["correct_predictions"] = (
        summary["true_positives"]
        + summary["true_negatives"]
    )

    # IMPORTANT:
    # Error rate is calculated using the 715 records
    # belonging to each individual experiment.
    summary["error_rate"] = (
        summary["total_errors"]
        / summary["total_records"]
    ) * 100

    summary["accuracy_from_predictions"] = (
        summary["correct_predictions"]
        / summary["total_records"]
    ) * 100

    # =========================================================
    # PRINT SUMMARY
    # =========================================================

    print()
    print("=" * 75)
    print("FALSE NEGATIVE / FALSE POSITIVE COUNTS")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # =========================================================
    # SCORE DISTANCE SUMMARY
    # =========================================================

    print()
    print("=" * 75)
    print("SCORE DISTANCE FROM 0.5 THRESHOLD")
    print("=" * 75)

    distance_summary = (
        df.groupby(
            "experiment"
        )[
            "score_distance_from_threshold"
        ]
        .agg(
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    print(
        distance_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =========================================================
    # SAVE SUMMARY
    # =========================================================

    summary_path = (
        report_dir
        / "error_analysis_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================
    # ADDITIONAL DATA INTEGRITY CHECK
    # =========================================================

    expected_rows_per_experiment = 715

    for experiment in expected_experiments:

        experiment_rows = df[
            df["experiment"] == experiment
        ]

        if len(experiment_rows) != (
            expected_rows_per_experiment
        ):
            raise ValueError(
                f"{experiment} contains "
                f"{len(experiment_rows)} rows; "
                f"expected "
                f"{expected_rows_per_experiment}."
            )

    if len(df) != 2145:
        raise ValueError(
            f"Expected 2145 prediction rows, "
            f"found {len(df)}."
        )

    # =========================================================
    # FILE OUTPUT
    # =========================================================

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(error_counts_path)
    print(subreddit_path)
    print(score_path)
    print(length_path)
    print(fn_path)
    print(fp_path)
    print(summary_path)

    print()
    print("=" * 75)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()