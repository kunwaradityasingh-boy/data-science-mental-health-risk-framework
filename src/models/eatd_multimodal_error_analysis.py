from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "eatd_multimodal_validation_predictions.csv"
)

THRESHOLD_RESULTS_PATH = (
    ROOT
    / "reports"
    / "eatd_multimodal_threshold_analysis.csv"
)

PROBABILITY_SUMMARY_PATH = (
    ROOT
    / "reports"
    / "eatd_multimodal_probability_summary.csv"
)

ERRORS_PATH = (
    ROOT
    / "reports"
    / "eatd_multimodal_errors.csv"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "eatd_multimodal_error_analysis_report.txt"
)


def calculate_metrics(y_true, probability, threshold):

    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
    }


def main():

    print("\n" + "=" * 75)
    print("EATD MULTIMODAL ERROR + THRESHOLD ANALYSIS")
    print("=" * 75)

    # =========================================================
    # 1. LOAD CURRENT FINAL FUSION PREDICTIONS
    # =========================================================

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    required_columns = {
        "participant_key",
        "split",
        "participant",
        "target",
        "text_probability",
        "voice_probability",
        "fusion_probability",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    # =========================================================
    # 2. VALIDATION CHECKS
    # =========================================================

    if len(df) != 79:
        raise ValueError(
            f"Expected 79 validation records, "
            f"found {len(df)}."
        )

    if df["participant_key"].duplicated().any():
        raise ValueError(
            "Duplicate participant_key detected."
        )

    if not df["split"].astype(str).str.lower().eq(
        "validation"
    ).all():
        raise ValueError(
            "Prediction file contains non-validation records."
        )

    y_true = (
        df["target"]
        .astype(int)
        .to_numpy()
    )

    fusion_probability = (
        df["fusion_probability"]
        .astype(float)
        .to_numpy()
    )

    # =========================================================
    # 3. DATA SUMMARY
    # =========================================================

    non_depressed = int(
        (y_true == 0).sum()
    )

    depressed = int(
        (y_true == 1).sum()
    )

    print("\nVALIDATION")
    print(f"Participants: {len(df)}")
    print(f"Non-depressed: {non_depressed}")
    print(f"Depressed: {depressed}")

    # =========================================================
    # 4. THRESHOLD ANALYSIS
    # =========================================================

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    threshold_results = []

    for threshold in thresholds:

        threshold_results.append(
            calculate_metrics(
                y_true,
                fusion_probability,
                threshold,
            )
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    # =========================================================
    # 5. PRIMARY THRESHOLD
    # =========================================================

    primary_threshold = 0.50

    primary_prediction = (
        fusion_probability
        >= primary_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        primary_prediction,
    )

    precision = precision_score(
        y_true,
        primary_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        primary_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        primary_prediction,
        zero_division=0,
    )

    roc_auc = (
        __import__(
            "sklearn.metrics",
            fromlist=["roc_auc_score"],
        )
        .roc_auc_score(
            y_true,
            fusion_probability,
        )
    )

    pr_auc = average_precision_score(
        y_true,
        fusion_probability,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            primary_prediction,
            labels=[0, 1],
        ).ravel()
    )

    # =========================================================
    # 6. ERROR CLASSIFICATION
    # =========================================================

    error_type = np.select(
        [
            (y_true == 0)
            & (primary_prediction == 0),

            (y_true == 0)
            & (primary_prediction == 1),

            (y_true == 1)
            & (primary_prediction == 0),

            (y_true == 1)
            & (primary_prediction == 1),
        ],
        [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        default="unknown",
    )

    errors_df = pd.DataFrame(
        {
            "participant_key":
                df["participant_key"].to_numpy(),

            "split":
                df["split"].to_numpy(),

            "participant":
                df["participant"].to_numpy(),

            "target":
                y_true,

            "text_probability":
                df["text_probability"]
                .astype(float)
                .to_numpy(),

            "voice_probability":
                df["voice_probability"]
                .astype(float)
                .to_numpy(),

            "fusion_probability":
                fusion_probability,

            "fusion_prediction":
                primary_prediction,

            "error_type":
                error_type,
        }
    )

    errors_df.to_csv(
        ERRORS_PATH,
        index=False,
    )

    # =========================================================
    # 7. PROBABILITY SUMMARY
    # =========================================================

    probability_summary = pd.DataFrame(
        [
            {
                "model":
                    "text_voice_fusion",

                "minimum":
                    float(
                        fusion_probability.min()
                    ),

                "mean":
                    float(
                        fusion_probability.mean()
                    ),

                "median":
                    float(
                        np.median(
                            fusion_probability
                        )
                    ),

                "maximum":
                    float(
                        fusion_probability.max()
                    ),

                "roc_auc":
                    float(roc_auc),

                "pr_auc":
                    float(pr_auc),
            }
        ]
    )

    probability_summary.to_csv(
        PROBABILITY_SUMMARY_PATH,
        index=False,
    )

    # =========================================================
    # 8. REPORT
    # =========================================================

    report = f"""
EATD MULTIMODAL ERROR + THRESHOLD ANALYSIS
============================================================

VALIDATION
----------
Participants: {len(df)}
Non-depressed: {non_depressed}
Depressed: {depressed}

PRIMARY THRESHOLD
-----------------
{primary_threshold:.2f}

PRIMARY METRICS
---------------
Accuracy : {accuracy:.6f}
Precision: {precision:.6f}
Recall   : {recall:.6f}
F1       : {f1:.6f}
ROC-AUC  : {roc_auc:.6f}
PR-AUC   : {pr_auc:.6f}

CONFUSION MATRIX
----------------
[[{tn} {fp}]
 [{fn} {tp}]]

ERROR COUNTS
------------
TN: {tn}
FP: {fp}
FN: {fn}
TP: {tp}

PROBABILITY SUMMARY
-------------------
Minimum : {fusion_probability.min():.6f}
Mean    : {fusion_probability.mean():.6f}
Median  : {np.median(fusion_probability):.6f}
Maximum : {fusion_probability.max():.6f}

METHODOLOGICAL NOTE
-------------------
The official EATD validation split was retained
as the final evaluation set.

Threshold analysis is descriptive and does not
represent clinical threshold optimization.

The multimodal model is a research benchmark
and must not be interpreted as a clinical diagnosis
or clinical probability.
"""

    REPORT_PATH.write_text(
        report.strip(),
        encoding="utf-8",
    )

    # =========================================================
    # 9. CONSOLE OUTPUT
    # =========================================================

    print("\nPRIMARY THRESHOLD = 0.50")
    print(
        f"Accuracy : {accuracy:.6f}"
    )
    print(
        f"Precision: {precision:.6f}"
    )
    print(
        f"Recall   : {recall:.6f}"
    )
    print(
        f"F1       : {f1:.6f}"
    )
    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )
    print(
        f"PR-AUC   : {pr_auc:.6f}"
    )

    print("\nCONFUSION MATRIX")
    print(
        np.array(
            [
                [tn, fp],
                [fn, tp],
            ]
        )
    )

    print("\nERROR COUNTS")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print("\nOUTPUT FILES")
    print(f"[OK] {THRESHOLD_RESULTS_PATH}")
    print(f"[OK] {PROBABILITY_SUMMARY_PATH}")
    print(f"[OK] {ERRORS_PATH}")
    print(f"[OK] {REPORT_PATH}")

    print("\nPHASE 8 STEP 3: PASSED")


if __name__ == "__main__":
    main()