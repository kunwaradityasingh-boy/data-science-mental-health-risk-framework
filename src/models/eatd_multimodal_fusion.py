from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = (
    ROOT / "data" / "processed" / "eatd"
    / "multimodal_oof_predictions.csv"
)

TEXT_MODEL_PATH = (
    ROOT / "models" / "eatd_text"
    / "tfidf_logistic_regression.joblib"
)

VOICE_MODEL_PATH = (
    ROOT / "models" / "eatd_voice"
    / "logistic_regression.joblib"
)

TEXT_DATA_PATH = (
    ROOT / "data" / "processed" / "eatd"
    / "eatd_text_data.csv"
)

ACOUSTIC_PATH = (
    ROOT / "data" / "processed" / "eatd"
    / "acoustic_features.csv"
)

MODEL_DIR = ROOT / "models" / "eatd_multimodal"
REPORT_DIR = ROOT / "reports"

OUTPUT_MODEL = (
    MODEL_DIR
    / "text_voice_fusion_logistic_regression.joblib"
)

RESULTS_PATH = (
    REPORT_DIR
    / "eatd_multimodal_fusion_results.csv"
)

CM_PATH = (
    REPORT_DIR
    / "eatd_multimodal_fusion_confusion_matrix.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "eatd_multimodal_validation_predictions.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "eatd_multimodal_fusion_report.txt"
)


def make_participant_key(split_series, participant_series):
    return (
        split_series.astype(str).str.strip()
        + "::"
        + participant_series.astype(str).str.strip()
    )


def calculate_metrics(y_true, probabilities, threshold=0.50):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }


def main():

    print("\n" + "=" * 75)
    print("EATD TEXT + VOICE MULTIMODAL FUSION")
    print("=" * 75)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # 1. LOAD OOF PREDICTIONS
    # =========================================================

    oof = pd.read_csv(
        OOF_PATH
    )

    required_columns = {
        "participant_key",
        "split",
        "participant",
        "target",
        "text_oof_probability",
        "voice_oof_probability",
    }

    missing = (
        required_columns
        - set(oof.columns)
    )

    if missing:
        raise ValueError(
            "Missing required OOF columns: "
            f"{sorted(missing)}"
        )

    train_oof = oof.loc[
        oof["split"]
        .astype(str)
        .str.lower()
        .eq("train")
    ].copy()

    if len(train_oof) != 83:
        raise ValueError(
            f"Expected 83 training OOF records, "
            f"found {len(train_oof)}."
        )

    if train_oof[
        "participant_key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate participant_key found "
            "in OOF predictions."
        )

    # =========================================================
    # 2. TRAIN FUSION MODEL ON OOF PREDICTIONS
    # =========================================================

    X_meta = train_oof[
        [
            "text_oof_probability",
            "voice_oof_probability",
        ]
    ].copy()

    y_meta = (
        train_oof["target"]
        .astype(int)
        .to_numpy()
    )

    fusion_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    fusion_model.fit(
        X_meta,
        y_meta,
    )

    joblib.dump(
        fusion_model,
        OUTPUT_MODEL,
    )

    # =========================================================
    # 3. LOAD FINAL TEXT MODEL
    # =========================================================

    text_model = joblib.load(
        TEXT_MODEL_PATH
    )

    text_df = pd.read_csv(
        TEXT_DATA_PATH
    )

    text_df["participant_key"] = (
        make_participant_key(
            text_df["split"],
            text_df["participant"],
        )
    )

    text_val = text_df.loc[
        text_df["split"]
        .astype(str)
        .str.lower()
        .eq("validation")
    ].copy()

    if len(text_val) != 79:
        raise ValueError(
            f"Expected 79 text validation "
            f"records, found {len(text_val)}."
        )

    text_probability = (
        text_model.predict_proba(
            text_val["text"]
        )[:, 1]
    )

    text_predictions = pd.DataFrame(
        {
            "participant_key":
                text_val["participant_key"].to_numpy(),

            "split":
                text_val["split"].to_numpy(),

            "participant":
                text_val["participant"].to_numpy(),

            "target":
                text_val["target"]
                .astype(int)
                .to_numpy(),

            "text_probability":
                text_probability,
        }
    )

    # =========================================================
    # 4. LOAD FINAL VOICE MODEL
    # =========================================================

    voice_model = joblib.load(
        VOICE_MODEL_PATH
    )

    acoustic = pd.read_csv(
        ACOUSTIC_PATH
    )

    # IMPORTANT:
    # Do NOT use assign() here.
    # Build the key separately to avoid fragmented
    # DataFrame warnings.

    voice_key = make_participant_key(
        acoustic["split"],
        acoustic["participant"],
    )

    validation_mask = (
        acoustic["split"]
        .astype(str)
        .str.lower()
        .eq("validation")
    )

    voice_val = acoustic.loc[
        validation_mask
    ].copy()

    voice_val_key = voice_key.loc[
        validation_mask
    ].to_numpy()

    if len(voice_val) != 79:
        raise ValueError(
            f"Expected 79 voice validation "
            f"records, found {len(voice_val)}."
        )

    metadata_columns = {
        "split",
        "participant",
        "participant_id",
        "label",
        "new_label",
        "target",
        "usable_responses",
    }

    voice_feature_columns = [
        column
        for column in voice_val.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(
            voice_val[column]
        )
    ]

    if len(voice_feature_columns) != 142:
        raise ValueError(
            f"Expected 142 acoustic features, "
            f"found {len(voice_feature_columns)}."
        )

    X_voice = voice_val[
        voice_feature_columns
    ].copy()

    # Preserve feature names exactly as used
    # when the voice model was trained.

    voice_probability = (
        voice_model.predict_proba(
            X_voice
        )[:, 1]
    )

    voice_predictions = pd.DataFrame(
        {
            "participant_key":
                voice_val_key,

            "voice_probability":
                voice_probability,
        }
    )

    # =========================================================
    # 5. ALIGN TEXT + VOICE
    # =========================================================

    validation = pd.merge(
        text_predictions,
        voice_predictions,
        on="participant_key",
        how="inner",
        validate="one_to_one",
    )

    if len(validation) != 79:
        raise ValueError(
            "Text/voice validation alignment "
            "failed."
        )

    # =========================================================
    # 6. PREPARE FUSION INPUT
    # =========================================================

    # CRITICAL:
    # The fusion model was FIT with these exact names:
    #
    # text_oof_probability
    # voice_oof_probability
    #
    # Therefore validation inputs must use
    # exactly the same names.

    X_fusion_validation = pd.DataFrame(
        {
            "text_oof_probability":
                validation[
                    "text_probability"
                ].to_numpy(),

            "voice_oof_probability":
                validation[
                    "voice_probability"
                ].to_numpy(),
        }
    )

    # =========================================================
    # 7. FUSION PREDICTION
    # =========================================================

    fusion_probability = (
        fusion_model.predict_proba(
            X_fusion_validation
        )[:, 1]
    )

    validation_predictions = pd.DataFrame(
        {
            "participant_key":
                validation[
                    "participant_key"
                ].to_numpy(),

            "split":
                validation["split"].to_numpy(),

            "participant":
                validation[
                    "participant"
                ].to_numpy(),

            "target":
                validation["target"]
                .astype(int)
                .to_numpy(),

            "text_probability":
                validation[
                    "text_probability"
                ].to_numpy(),

            "voice_probability":
                validation[
                    "voice_probability"
                ].to_numpy(),

            "fusion_probability":
                fusion_probability,
        }
    )

    # =========================================================
    # 8. METRICS
    # =========================================================

    y_true = (
        validation_predictions[
            "target"
        ].to_numpy()
    )

    threshold = 0.50

    result = calculate_metrics(
        y_true,
        fusion_probability,
        threshold,
    )

    predictions = (
        fusion_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    # =========================================================
    # 9. SAVE VALIDATION PREDICTIONS
    # =========================================================

    validation_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # =========================================================
    # 10. SAVE RESULTS
    # =========================================================

    pd.DataFrame(
        [
            {
                "model":
                    "text_voice_fusion_logistic_regression",

                "validation_participants":
                    len(y_true),

                "non_depressed":
                    int((y_true == 0).sum()),

                "depressed":
                    int((y_true == 1).sum()),

                "threshold":
                    threshold,

                **result,
            }
        ]
    ).to_csv(
        RESULTS_PATH,
        index=False,
    )

    # =========================================================
    # 11. SAVE CONFUSION MATRIX
    # =========================================================

    pd.DataFrame(
        [
            {
                "actual":
                    "non_depressed",

                "predicted":
                    "non_depressed",

                "count":
                    tn,
            },
            {
                "actual":
                    "non_depressed",

                "predicted":
                    "depressed",

                "count":
                    fp,
            },
            {
                "actual":
                    "depressed",

                "predicted":
                    "non_depressed",

                "count":
                    fn,
            },
            {
                "actual":
                    "depressed",

                "predicted":
                    "depressed",

                "count":
                    tp,
            },
        ]
    ).to_csv(
        CM_PATH,
        index=False,
    )

    # =========================================================
    # 12. SAVE REPORT
    # =========================================================

    report = f"""
EATD TEXT + VOICE MULTIMODAL FUSION REPORT
============================================================

DATA
----
Training participants: 83
Validation participants: 79
Validation non-depressed: {(y_true == 0).sum()}
Validation depressed: {(y_true == 1).sum()}

FUSION METHOD
-------------
Meta-model: Logistic Regression
Meta-model inputs:
- Text OOF probability
- Voice OOF probability

OOF folds: 5

Official EATD validation split remained
untouched during meta-model training.

Final text and voice models were trained
on the complete training split before
validation prediction.

Split-aware participant keys were used.

PRIMARY THRESHOLD
-----------------
0.50

RESULTS
-------
Accuracy : {result["accuracy"]:.6f}
Precision: {result["precision"]:.6f}
Recall   : {result["recall"]:.6f}
F1       : {result["f1"]:.6f}
ROC-AUC  : {result["roc_auc"]:.6f}
PR-AUC   : {result["pr_auc"]:.6f}

CONFUSION MATRIX
----------------
TN: {tn}
FP: {fp}
FN: {fn}
TP: {tp}

REPRODUCIBILITY
---------------
Random state: 42
Official train/validation split preserved.
No validation labels used for meta-model fitting.
No validation threshold tuning performed.

LIMITATION
----------
The multimodal validation result is a research
benchmark and must not be interpreted as a
clinical diagnosis or clinical probability.
"""

    REPORT_PATH.write_text(
        report.strip(),
        encoding="utf-8",
    )

    # =========================================================
    # 13. FINAL CONSOLE OUTPUT
    # =========================================================

    print("\nDATA")
    print(
        f"Training participants   : {len(train_oof)}"
    )
    print(
        f"Validation participants: {len(y_true)}"
    )

    print("\nVALIDATION CLASS DISTRIBUTION")
    print(
        f"Non-depressed: {(y_true == 0).sum()}"
    )
    print(
        f"Depressed    : {(y_true == 1).sum()}"
    )

    print("\nFINAL FUSION RESULTS")
    print(
        f"Accuracy : {result['accuracy']:.6f}"
    )
    print(
        f"Precision: {result['precision']:.6f}"
    )
    print(
        f"Recall   : {result['recall']:.6f}"
    )
    print(
        f"F1       : {result['f1']:.6f}"
    )
    print(
        f"ROC-AUC  : {result['roc_auc']:.6f}"
    )
    print(
        f"PR-AUC   : {result['pr_auc']:.6f}"
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

    print("\nOUTPUT FILES")
    print(f"[OK] {OUTPUT_MODEL}")
    print(f"[OK] {RESULTS_PATH}")
    print(f"[OK] {CM_PATH}")
    print(f"[OK] {PREDICTIONS_PATH}")
    print(f"[OK] {REPORT_PATH}")

    print("\nPHASE 8 STEP 2: PASSED")


if __name__ == "__main__":
    main()