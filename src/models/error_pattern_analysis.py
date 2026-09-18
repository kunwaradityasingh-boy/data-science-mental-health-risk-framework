from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]
    report_dir = base_dir / "reports"

    prediction_path = (
        report_dir / "final_holdout_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {prediction_path}"
        )

    df = pd.read_csv(prediction_path)

    expected_experiments = {
        "text_only",
        "text_plus_social",
        "text_plus_all_engineered",
    }

    # =========================================================
    # BASIC INFORMATION
    # =========================================================

    print("=" * 75)
    print("DREAddit HOLDOUT ERROR PATTERN ANALYSIS")
    print("=" * 75)

    print(f"Rows loaded : {len(df)}")

    # =========================================================
    # ERROR FLAG
    # =========================================================

    df["is_error"] = df["error_type"].isin(
        [
            "false_positive",
            "false_negative",
        ]
    )

    # =========================================================
    # 1. SUBREDDIT ERROR ANALYSIS
    # =========================================================

    subreddit = (
        df.groupby(
            [
                "experiment",
                "subreddit",
            ]
        )
        .agg(
            records=("id", "count"),
            errors=("is_error", "sum"),
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

    subreddit["error_rate"] = (
        subreddit["errors"]
        / subreddit["records"]
    ) * 100

    subreddit["error_rate"] = subreddit[
        "error_rate"
    ].round(2)

    subreddit = subreddit.sort_values(
        [
            "experiment",
            "error_rate",
        ],
        ascending=[
            True,
            False,
        ],
    )

    subreddit_path = (
        report_dir
        / "error_pattern_subreddit_summary.csv"
    )

    subreddit.to_csv(
        subreddit_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("SUBREDDIT ERROR PATTERNS")
    print("=" * 75)

    for experiment in expected_experiments:

        print()
        print(f"--- {experiment} ---")

        subset = subreddit[
            subreddit["experiment"]
            == experiment
        ].copy()

        print(
            subset[
                [
                    "subreddit",
                    "records",
                    "errors",
                    "false_positives",
                    "false_negatives",
                    "error_rate",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # 2. SCORE BAND ANALYSIS
    # =========================================================

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

    score = (
        df.groupby(
            [
                "experiment",
                "score_band",
            ],
            observed=True,
        )
        .agg(
            records=("id", "count"),
            errors=("is_error", "sum"),
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

    score["error_rate"] = (
        score["errors"]
        / score["records"]
    ) * 100

    score["error_rate"] = score[
        "error_rate"
    ].round(2)

    score_path = (
        report_dir
        / "error_pattern_score_summary.csv"
    )

    score.to_csv(
        score_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("PREDICTION SCORE BAND PATTERNS")
    print("=" * 75)

    for experiment in expected_experiments:

        print()
        print(f"--- {experiment} ---")

        subset = score[
            score["experiment"]
            == experiment
        ]

        print(
            subset.to_string(
                index=False
            )
        )

    # =========================================================
    # 3. DISTANCE FROM DECISION THRESHOLD
    # =========================================================

    df["distance_from_0_5"] = (
        df["class_1_score"] - 0.5
    ).abs()

    threshold_summary = (
        df.groupby("experiment")
        .agg(
            mean_distance=(
                "distance_from_0_5",
                "mean",
            ),
            median_distance=(
                "distance_from_0_5",
                "median",
            ),
            error_mean_distance=(
                "distance_from_0_5",
                lambda x: x[
                    df.loc[
                        x.index,
                        "is_error",
                    ]
                ].mean(),
            ),
            correct_mean_distance=(
                "distance_from_0_5",
                lambda x: x[
                    ~df.loc[
                        x.index,
                        "is_error",
                    ]
                ].mean(),
            ),
        )
        .reset_index()
    )

    threshold_summary[
        [
            "mean_distance",
            "median_distance",
            "error_mean_distance",
            "correct_mean_distance",
        ]
    ] = threshold_summary[
        [
            "mean_distance",
            "median_distance",
            "error_mean_distance",
            "correct_mean_distance",
        ]
    ].round(4)

    threshold_path = (
        report_dir
        / "error_pattern_threshold_summary.csv"
    )

    threshold_summary.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("DECISION THRESHOLD DISTANCE")
    print("=" * 75)

    print(
        threshold_summary.to_string(
            index=False
        )
    )

    # =========================================================
    # 4. TEXT LENGTH ANALYSIS
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

    length = (
        df.groupby(
            [
                "experiment",
                "text_length_band",
            ],
            observed=True,
        )
        .agg(
            records=("id", "count"),
            errors=("is_error", "sum"),
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

    length["error_rate"] = (
        length["errors"]
        / length["records"]
    ) * 100

    length["error_rate"] = length[
        "error_rate"
    ].round(2)

    length_path = (
        report_dir
        / "error_pattern_text_length_summary.csv"
    )

    length.to_csv(
        length_path,
        index=False,
    )

    print()
    print("=" * 75)
    print("TEXT LENGTH ERROR PATTERNS")
    print("=" * 75)

    for experiment in expected_experiments:

        print()
        print(f"--- {experiment} ---")

        subset = length[
            length["experiment"]
            == experiment
        ]

        print(
            subset.to_string(
                index=False
            )
        )

    # =========================================================
    # 5. FALSE NEGATIVE ANALYSIS
    # =========================================================

    fn = df[
        df["error_type"]
        == "false_negative"
    ].copy()

    fn_summary = (
        fn.groupby("experiment")
        .agg(
            false_negatives=("id", "count"),
            mean_score=(
                "class_1_score",
                "mean",
            ),
            median_score=(
                "class_1_score",
                "median",
            ),
            mean_text_length=(
                "text_length",
                "mean",
            ),
        )
        .reset_index()
    )

    fn_summary[
        [
            "mean_score",
            "median_score",
            "mean_text_length",
        ]
    ] = fn_summary[
        [
            "mean_score",
            "median_score",
            "mean_text_length",
        ]
    ].round(4)

    fn_summary_path = (
        report_dir
        / "error_pattern_false_negative_summary.csv"
    )

    fn_summary.to_csv(
        fn_summary_path,
        index=False,
    )

    # =========================================================
    # 6. FALSE POSITIVE ANALYSIS
    # =========================================================

    fp = df[
        df["error_type"]
        == "false_positive"
    ].copy()

    fp_summary = (
        fp.groupby("experiment")
        .agg(
            false_positives=("id", "count"),
            mean_score=(
                "class_1_score",
                "mean",
            ),
            median_score=(
                "class_1_score",
                "median",
            ),
            mean_text_length=(
                "text_length",
                "mean",
            ),
        )
        .reset_index()
    )

    fp_summary[
        [
            "mean_score",
            "median_score",
            "mean_text_length",
        ]
    ] = fp_summary[
        [
            "mean_score",
            "median_score",
            "mean_text_length",
        ]
    ].round(4)

    fp_summary_path = (
        report_dir
        / "error_pattern_false_positive_summary.csv"
    )

    fp_summary.to_csv(
        fp_summary_path,
        index=False,
    )

    # =========================================================
    # 7. OVERALL ERROR SUMMARY
    # =========================================================

    overall = (
        df.groupby("experiment")
        .agg(
            records=("id", "count"),
            errors=("is_error", "sum"),
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

    overall["error_rate"] = (
        overall["errors"]
        / overall["records"]
    ) * 100

    overall["error_rate"] = overall[
        "error_rate"
    ].round(2)

    overall_path = (
        report_dir
        / "error_pattern_overall_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    print()
    print("=" * 75)
    print("OVERALL ERROR SUMMARY")
    print("=" * 75)

    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print("=" * 75)
    print("ANALYSIS FILES SAVED")
    print("=" * 75)

    print(subreddit_path)
    print(score_path)
    print(threshold_path)
    print(length_path)
    print(fn_summary_path)
    print(fp_summary_path)
    print(overall_path)

    print()
    print("=" * 75)
    print("ERROR PATTERN ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()