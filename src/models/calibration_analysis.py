from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    prediction_path = (
        base_dir
        / "reports"
        / "final_holdout_predictions.csv"
    )

    report_dir = base_dir / "reports"

    df = pd.read_csv(prediction_path)

    experiments = [
        "text_only",
        "text_plus_social",
        "text_plus_all_engineered",
    ]

    print("=" * 75)
    print("DREAddit FINAL HOLDOUT CALIBRATION ANALYSIS")
    print("=" * 75)

    print(f"Rows loaded : {len(df)}")

    results = []

    calibration_rows = []

    for experiment in experiments:

        subset = df[
            df["experiment"] == experiment
        ].copy()

        y_true = subset[
            "actual_label"
        ].astype(int).to_numpy()

        y_score = subset[
            "class_1_score"
        ].astype(float).to_numpy()

        # -----------------------------------------------------
        # Brier score
        # -----------------------------------------------------

        brier = brier_score_loss(
            y_true,
            y_score,
        )

        # -----------------------------------------------------
        # Log loss
        # -----------------------------------------------------

        logloss = log_loss(
            y_true,
            y_score,
        )

        # -----------------------------------------------------
        # Calibration curve
        # -----------------------------------------------------

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_true,
                y_score,
                n_bins=10,
                strategy="uniform",
            )
        )

        # -----------------------------------------------------
        # Calibration error
        # -----------------------------------------------------

        calibration_error = np.mean(
            np.abs(
                fraction_positive
                - mean_predicted
            )
        )

        results.append(
            {
                "experiment": experiment,
                "records": len(subset),
                "brier_score": brier,
                "log_loss": logloss,
                "mean_absolute_calibration_error":
                    calibration_error,
            }
        )

        for bin_number, (
            predicted,
            observed,
        ) in enumerate(
            zip(
                mean_predicted,
                fraction_positive,
            ),
            start=1,
        ):

            calibration_rows.append(
                {
                    "experiment": experiment,
                    "bin": bin_number,
                    "mean_predicted_score": predicted,
                    "observed_positive_fraction": observed,
                    "absolute_difference": abs(
                        predicted - observed
                    ),
                }
            )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    summary = pd.DataFrame(results)

    calibration_detail = pd.DataFrame(
        calibration_rows
    )

    summary_path = (
        report_dir
        / "calibration_summary.csv"
    )

    detail_path = (
        report_dir
        / "calibration_bins.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    calibration_detail.to_csv(
        detail_path,
        index=False,
    )

    # =========================================================
    # PRINT SUMMARY
    # =========================================================

    print()
    print("=" * 75)
    print("CALIBRATION SUMMARY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 75)
    print("CALIBRATION BIN DETAILS")
    print("=" * 75)

    print(
        calibration_detail.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(summary_path)
    print(detail_path)

    print()
    print("=" * 75)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()