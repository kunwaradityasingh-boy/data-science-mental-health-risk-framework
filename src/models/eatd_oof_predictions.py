from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
    / "eatd_text_data.csv"
)

VOICE_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
    / "acoustic_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_OUTPUT = OUTPUT_DIR / "multimodal_oof_predictions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# HELPER
# ============================================================

def make_key(df):
    """
    Creates a split-aware participant key.

    Numeric participant IDs can occur in both official EATD
    split directories, therefore participant alone must NOT
    be used as the merge key.
    """

    return (
        df["split"].astype(str).str.lower()
        + "::"
        + df["participant"].astype(str)
    )


# ============================================================
# LOAD DATA
# ============================================================

if not TEXT_DATA.exists():
    raise FileNotFoundError(
        f"Text dataset not found: {TEXT_DATA}"
    )

if not VOICE_DATA.exists():
    raise FileNotFoundError(
        f"Voice dataset not found: {VOICE_DATA}"
    )


text_df = pd.read_csv(TEXT_DATA)
voice_df = pd.read_csv(VOICE_DATA)


# ============================================================
# TARGET
# ============================================================

text_df["target"] = (
    pd.to_numeric(text_df["new_label"], errors="coerce") > 52
).astype(int)

voice_df["target"] = (
    pd.to_numeric(voice_df["new_label"], errors="coerce") > 52
).astype(int)


# ============================================================
# SPLIT NORMALIZATION
# ============================================================

text_df["split"] = (
    text_df["split"]
    .astype(str)
    .str.lower()
)

voice_df["split"] = (
    voice_df["split"]
    .astype(str)
    .str.lower()
)


# ============================================================
# SPLIT-AWARE KEYS
# ============================================================

text_df["participant_key"] = make_key(text_df)
voice_df["participant_key"] = make_key(voice_df)


# ============================================================
# BASIC VALIDATION
# ============================================================

if text_df["participant_key"].duplicated().any():
    duplicates = text_df.loc[
        text_df["participant_key"].duplicated(),
        "participant_key"
    ].tolist()

    raise ValueError(
        "Duplicate text participant keys detected:\n"
        + "\n".join(map(str, duplicates[:20]))
    )


if voice_df["participant_key"].duplicated().any():
    duplicates = voice_df.loc[
        voice_df["participant_key"].duplicated(),
        "participant_key"
    ].tolist()

    raise ValueError(
        "Duplicate voice participant keys detected:\n"
        + "\n".join(map(str, duplicates[:20]))
    )


# ============================================================
# MERGE TEXT + VOICE METADATA
# ============================================================

text_meta = text_df[
    [
        "participant_key",
        "split",
        "participant",
        "new_label",
        "target",
        "text",
    ]
].copy()

voice_meta = voice_df[
    [
        "participant_key",
        "split",
        "participant",
        "new_label",
        "target",
    ]
].copy()


# Keep all acoustic feature columns.
voice_feature_columns = [
    c
    for c in voice_df.columns
    if c.startswith("acoustic_")
]


# Fallback for the actual current feature file:
# select numeric feature columns excluding metadata.
if not voice_feature_columns:

    excluded = {
        "target",
        "label",
        "new_label",
        "participant",
        "split",
        "usable_responses",
    }

    voice_feature_columns = [
        c
        for c in voice_df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            voice_df[c]
        )
    ]


if not voice_feature_columns:
    raise ValueError(
        "No voice acoustic feature columns found."
    )


# ============================================================
# TRAIN / VALIDATION SPLITS
# ============================================================

train_text = text_df[
    text_df["split"] == "train"
].copy()

validation_text = text_df[
    text_df["split"] == "validation"
].copy()

train_voice = voice_df[
    voice_df["split"] == "train"
].copy()

validation_voice = voice_df[
    voice_df["split"] == "validation"
].copy()


# ============================================================
# MATCH TRAIN PARTICIPANTS
# ============================================================

train_keys = sorted(
    set(train_text["participant_key"])
    & set(train_voice["participant_key"])
)

validation_keys = sorted(
    set(validation_text["participant_key"])
    & set(validation_voice["participant_key"])
)


if len(train_keys) != len(train_text):
    missing_voice = sorted(
        set(train_text["participant_key"])
        - set(train_voice["participant_key"])
    )

    raise ValueError(
        "Training text/voice participants do not match.\n"
        f"Missing voice records: {missing_voice[:20]}"
    )


if len(validation_keys) != len(validation_voice):
    missing_text = sorted(
        set(validation_voice["participant_key"])
        - set(validation_text["participant_key"])
    )

    raise ValueError(
        "Validation text/voice participants do not match.\n"
        f"Missing text records: {missing_text[:20]}"
    )


# ============================================================
# BUILD TRAINING FUSION TABLE
# ============================================================

train_text_indexed = train_text.set_index(
    "participant_key"
).loc[train_keys]

train_voice_indexed = train_voice.set_index(
    "participant_key"
).loc[train_keys]


train_fusion = pd.DataFrame(
    {
        "participant_key": train_keys,
        "split": "train",
        "participant": train_text_indexed[
            "participant"
        ].to_numpy(),
        "target": train_text_indexed[
            "target"
        ].astype(int).to_numpy(),
        "text": train_text_indexed[
            "text"
        ].fillna("").astype(str).to_numpy(),
    }
)

voice_matrix = train_voice_indexed[
    voice_feature_columns
].copy()

for column in voice_matrix.columns:
    voice_matrix[column] = pd.to_numeric(
        voice_matrix[column],
        errors="coerce"
    )

voice_matrix = voice_matrix.to_numpy(
    dtype=float
)


# ============================================================
# STRATIFIED OOF SETUP
# ============================================================

y_train = train_fusion["target"].to_numpy()

class_counts = np.bincount(y_train)

if len(class_counts) < 2:
    raise ValueError(
        "Training data contains only one class."
    )

minimum_class_count = int(
    class_counts.min()
)

if minimum_class_count < N_SPLITS:
    raise ValueError(
        f"Not enough samples in minority class for "
        f"{N_SPLITS}-fold OOF CV. "
        f"Minimum class count={minimum_class_count}."
    )


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


text_oof = np.full(
    len(train_fusion),
    np.nan,
    dtype=float
)

voice_oof = np.full(
    len(train_fusion),
    np.nan,
    dtype=float
)


# ============================================================
# OOF LOOP
# ============================================================

for fold, (train_idx, valid_idx) in enumerate(
    skf.split(
        train_fusion,
        y_train
    ),
    start=1
):

    print()
    print(
        f"========== FOLD {fold}/{N_SPLITS} =========="
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    X_text_train = train_fusion.iloc[
        train_idx
    ]["text"].fillna("").astype(str)

    X_text_valid = train_fusion.iloc[
        valid_idx
    ]["text"].fillna("").astype(str)

    y_fold_train = y_train[train_idx]

    text_model.fit(
        X_text_train,
        y_fold_train
    )

    text_oof[valid_idx] = (
        text_model.predict_proba(
            X_text_valid
        )[:, 1]
    )

    # --------------------------------------------------------
    # VOICE
    # --------------------------------------------------------

    voice_model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    X_voice_train = voice_matrix[
        train_idx
    ]

    X_voice_valid = voice_matrix[
        valid_idx
    ]

    voice_model.fit(
        X_voice_train,
        y_fold_train
    )

    voice_oof[valid_idx] = (
        voice_model.predict_proba(
            X_voice_valid
        )[:, 1]
    )

    print(
        f"Fold {fold} text OOF predictions: "
        f"{len(valid_idx)}"
    )

    print(
        f"Fold {fold} voice OOF predictions: "
        f"{len(valid_idx)}"
    )


# ============================================================
# OOF VALIDATION
# ============================================================

if np.isnan(text_oof).any():
    raise RuntimeError(
        "Text OOF predictions contain NaN values."
    )

if np.isnan(voice_oof).any():
    raise RuntimeError(
        "Voice OOF predictions contain NaN values."
    )


text_oof_auc = roc_auc_score(
    y_train,
    text_oof
)

text_oof_pr_auc = average_precision_score(
    y_train,
    text_oof
)

voice_oof_auc = roc_auc_score(
    y_train,
    voice_oof
)

voice_oof_pr_auc = average_precision_score(
    y_train,
    voice_oof
)


# ============================================================
# SAVE OOF DATASET
# ============================================================

oof_df = train_fusion[
    [
        "participant_key",
        "split",
        "participant",
        "target",
    ]
].copy()

oof_df["text_oof_probability"] = text_oof

oof_df["voice_oof_probability"] = voice_oof


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

if len(oof_df) != len(train_text):
    raise RuntimeError(
        "OOF row count does not match training text records."
    )

if oof_df["participant_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate participant keys in OOF output."
    )


# ============================================================
# SAVE
# ============================================================

oof_df.to_csv(
    OOF_OUTPUT,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report_path = (
    PROJECT_ROOT
    / "reports"
    / "eatd_oof_prediction_report.txt"
)

report_lines = [
    "EATD MULTIMODAL OOF PREDICTION REPORT",
    "=" * 60,
    "",
    "Dataset: EATD-Corpus",
    "Modality: Text + Voice",
    "Target: new_label > 52",
    "Training split only: YES",
    f"Number of folds: {N_SPLITS}",
    f"Random state: {RANDOM_STATE}",
    "",
    "TRAINING DATA",
    "-" * 60,
    f"Participants: {len(oof_df)}",
    f"Non-depressed: {(y_train == 0).sum()}",
    f"Depressed: {(y_train == 1).sum()}",
    "",
    "TEXT OOF PERFORMANCE",
    "-" * 60,
    f"ROC-AUC: {text_oof_auc:.6f}",
    f"PR-AUC: {text_oof_pr_auc:.6f}",
    "",
    "VOICE OOF PERFORMANCE",
    "-" * 60,
    f"ROC-AUC: {voice_oof_auc:.6f}",
    f"PR-AUC: {voice_oof_pr_auc:.6f}",
    "",
    "LEAKAGE CONTROL",
    "-" * 60,
    "OOF probabilities generated using StratifiedKFold.",
    "Each OOF prediction was produced by a model that did not",
    "train on that participant.",
    "The official EATD validation split was not used.",
    "Text and voice records were joined using split-aware",
    "participant keys.",
    "",
    "STATUS",
    "-" * 60,
    "OOF prediction generation: PASSED",
    "Phase 8 Step 1: COMPLETE",
]

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EATD MULTIMODAL OOF PREDICTIONS")
print("=" * 70)

print(f"Training participants: {len(oof_df)}")
print(
    f"Training depressed: "
    f"{int((y_train == 1).sum())}"
)
print(
    f"Training non-depressed: "
    f"{int((y_train == 0).sum())}"
)

print()
print("TEXT OOF")
print(f"ROC-AUC: {text_oof_auc:.6f}")
print(f"PR-AUC : {text_oof_pr_auc:.6f}")

print()
print("VOICE OOF")
print(f"ROC-AUC: {voice_oof_auc:.6f}")
print(f"PR-AUC : {voice_oof_pr_auc:.6f}")

print()
print("OUTPUT")
print(f"[OK] {OOF_OUTPUT}")
print(f"[OK] {report_path}")

print()
print("PHASE 8 STEP 1: PASSED")