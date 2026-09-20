from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "phase10_error_analysis_summary.md"


def load_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {filename}: {exc}")
        return None


def clean_columns(df):
    if df is None:
        return None

    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def find_column(df, candidates):
    if df is None:
        return None

    columns = set(df.columns)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    for col in df.columns:
        for candidate in candidates:
            if candidate in col:
                return col

    return None


def basic_summary(df):
    if df is None or df.empty:
        return "No records available."

    return f"{len(df)} records."


def class_distribution(df):
    if df is None or df.empty:
        return None

    target_col = find_column(
        df,
        [
            "target",
            "label",
            "y_true",
            "true_label",
            "actual",
            "class",
        ],
    )

    if target_col is None:
        return None

    counts = df[target_col].value_counts(dropna=False)

    return counts.to_dict()


def error_counts(df):
    if df is None or df.empty:
        return None

    true_col = find_column(
        df,
        [
            "y_true",
            "true_label",
            "actual",
            "target",
            "label",
        ],
    )

    pred_col = find_column(
        df,
        [
            "y_pred",
            "predicted",
            "prediction",
            "pred_label",
        ],
    )

    if true_col is None or pred_col is None:
        return None

    true = pd.to_numeric(df[true_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")

    valid = true.notna() & pred.notna()

    true = true[valid]
    pred = pred[valid]

    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def probability_summary(df):
    if df is None or df.empty:
        return None

    probability_col = find_column(
        df,
        [
            "probability",
            "prob",
            "positive_probability",
            "class_1_probability",
            "score",
        ],
    )

    if probability_col is None:
        return None

    values = pd.to_numeric(
        df[probability_col],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def make_section(title, body):
    return f"## {title}\n\n{body}\n\n"


# ---------------------------------------------------------
# Load primary error-analysis files
# ---------------------------------------------------------

files = {
    "eatd_text_errors": "eatd_text_false_negatives.csv",
    "eatd_voice_errors": "eatd_voice_false_negatives.csv",
    "eatd_multimodal_errors": "eatd_multimodal_errors.csv",
    "dreaddit_summary": "error_analysis_summary.csv",
    "dreaddit_counts": "error_analysis_counts.csv",
    "dreaddit_score_bands": "error_analysis_score_bands.csv",
    "dreaddit_subreddit": "error_analysis_subreddit.csv",
    "dreaddit_text_length": "error_analysis_text_length.csv",
    "dreaddit_pattern_overall": "error_pattern_overall_summary.csv",
    "dreaddit_pattern_score": "error_pattern_score_summary.csv",
    "dreaddit_pattern_threshold": "error_pattern_threshold_summary.csv",
    "dreaddit_pattern_subreddit": "error_pattern_subreddit_summary.csv",
    "dreaddit_pattern_text_length": "error_pattern_text_length_summary.csv",
    "eatd_text_threshold": "eatd_text_threshold_analysis.csv",
    "eatd_voice_threshold": "eatd_voice_threshold_analysis.csv",
    "eatd_multimodal_threshold": "eatd_multimodal_threshold_analysis.csv",
    "eatd_multimodal_probability": "eatd_multimodal_probability_summary.csv",
}


data = {
    key: clean_columns(load_csv(filename))
    for key, filename in files.items()
}


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

report = ""

report += "# Phase 10.3 — Error Analysis Summary\n\n"
report += (
    "Project: Data Science Framework for Early Detection of "
    "Mental Health Risk\n\n"
)
report += "Framework Version: 0.1.0\n\n"

report += (
    "This report summarizes observed prediction errors, "
    "threshold behavior, class imbalance, and dataset-specific "
    "limitations across the evaluated experiments. The analysis "
    "is descriptive and does not establish clinical validity.\n\n"
)

# ---------------------------------------------------------
# EATD TEXT
# ---------------------------------------------------------

df = data["eatd_text_errors"]

report += make_section(
    "1. EATD Text Error Analysis",
    (
        f"Available false-negative records: {basic_summary(df)}\n\n"
        "The EATD text baseline showed high overall accuracy but "
        "failed to identify the positive/depression class reliably. "
        "The recorded validation result had precision = 0, recall = 0 "
        "and F1 = 0 for the positive class, while ROC-AUC was close "
        "to chance level. Therefore, the observed accuracy should not "
        "be interpreted as evidence of useful discrimination.\n\n"
        "The false-negative records are retained as an explicit "
        "error-analysis artifact rather than being treated as "
        "successful predictions."
    ),
)

# ---------------------------------------------------------
# EATD VOICE
# ---------------------------------------------------------

df = data["eatd_voice_errors"]

report += make_section(
    "2. EATD Voice Error Analysis",
    (
        f"Available false-negative records: {basic_summary(df)}\n\n"
        "The EATD voice Logistic Regression baseline produced "
        "limited positive-class detection. Validation recall and "
        "F1 for the positive class were low, and the ROC-AUC result "
        "was only modestly above or around chance depending on the "
        "evaluation view.\n\n"
        "The Random Forest baseline produced no positive predictions "
        "under the evaluated default threshold, resulting in zero "
        "positive-class recall and F1. This indicates that the "
        "classification threshold and class imbalance materially "
        "affect the observed operating behavior."
    ),
)

# ---------------------------------------------------------
# EATD MULTIMODAL
# ---------------------------------------------------------

df = data["eatd_multimodal_errors"]

report += make_section(
    "3. EATD Text + Voice Multimodal Error Analysis",
    (
        f"Available error records: {basic_summary(df)}\n\n"
        "The final EATD text + voice fusion experiment achieved "
        "accuracy = 0.7468, precision = 0.2222, recall = 0.1333, "
        "F1 = 0.1667, ROC-AUC = 0.5167 and PR-AUC = 0.2152.\n\n"
        "The confusion matrix was:\n\n"
        "| | Predicted 0 | Predicted 1 |\n"
        "| --- | ---: | ---: |\n"
        "| Actual 0 | 57 | 7 |\n"
        "| Actual 1 | 13 | 2 |\n\n"
        "The error pattern shows that most positive examples remained "
        "undetected at the evaluated operating threshold. Therefore, "
        "multimodal fusion did not demonstrate strong discrimination "
        "on this dataset under the tested protocol."
    ),
)

# ---------------------------------------------------------
# DREADDIT
# ---------------------------------------------------------

dreaddit_summary = data["dreaddit_summary"]
dreaddit_counts = data["dreaddit_counts"]
dreaddit_score = data["dreaddit_score_bands"]
dreaddit_subreddit = data["dreaddit_subreddit"]
dreaddit_length = data["dreaddit_text_length"]

dreaddit_body = (
    "Dreaddit provides the strongest classification evidence among "
    "the evaluated text experiments, but its errors still need to "
    "be interpreted in the context of the dataset and evaluation "
    "protocol.\n\n"
    "The final supplied holdout results for the text-only model were "
    "accuracy = 0.7175, F1 = 0.7363, ROC-AUC = 0.8207 and PR-AUC = "
    "0.8412. Adding engineered features improved the final holdout "
    "accuracy to 0.7594 and F1 to 0.7731, with ROC-AUC = 0.8422 and "
    "PR-AUC = 0.8513.\n\n"
    "Existing error-analysis files examine error counts, score bands, "
    "subreddit patterns and text-length patterns. These analyses "
    "should be interpreted as dataset-specific patterns rather than "
    "general clinical relationships."
)

if dreaddit_summary is not None:
    dreaddit_body += (
        f"\n\nSummary table records available: "
        f"{len(dreaddit_summary)}."
    )

if dreaddit_counts is not None:
    dreaddit_body += (
        f"\nError-count records available: "
        f"{len(dreaddit_counts)}."
    )

if dreaddit_score is not None:
    dreaddit_body += (
        f"\nScore-band records available: "
        f"{len(dreaddit_score)}."
    )

if dreaddit_subreddit is not None:
    dreaddit_body += (
        f"\nSubreddit-analysis records available: "
        f"{len(dreaddit_subreddit)}."
    )

if dreaddit_length is not None:
    dreaddit_body += (
        f"\nText-length-analysis records available: "
        f"{len(dreaddit_length)}."
    )

report += make_section("4. Dreaddit Error Analysis", dreaddit_body)

# ---------------------------------------------------------
# CLASS IMBALANCE
# ---------------------------------------------------------

report += make_section(
    "5. Class Imbalance",
    (
        "Class imbalance is particularly important for the EATD "
        "experiments. The EATD text dataset contains substantially "
        "more non-depression examples than depression examples. "
        "Consequently, a model can obtain relatively high accuracy "
        "while producing poor positive-class recall.\n\n"
        "This is directly reflected in the EATD text and voice "
        "experiments, where positive-class recall and F1 remained "
        "very low despite accuracy values around the mid-to-high "
        "70% range.\n\n"
        "For this reason, accuracy is not used as the sole indicator "
        "of model quality in this framework."
    ),
)

# ---------------------------------------------------------
# THRESHOLD ANALYSIS
# ---------------------------------------------------------

threshold_files = [
    data["eatd_text_threshold"],
    data["eatd_voice_threshold"],
    data["eatd_multimodal_threshold"],
]

threshold_records = sum(
    len(x) for x in threshold_files
    if x is not None
)

report += make_section(
    "6. Threshold Behavior",
    (
        f"Threshold-analysis records available across the EATD "
        f"experiments: {threshold_records}.\n\n"
        "Threshold analysis demonstrates that changing the "
        "classification threshold changes the precision/recall "
        "trade-off. However, threshold selection alone cannot "
        "convert a weakly discriminative model into a validated "
        "screening system.\n\n"
        "The framework therefore treats threshold selection as an "
        "evaluation and operating-point decision rather than as "
        "evidence of clinical performance."
    ),
)

# ---------------------------------------------------------
# ERROR PATTERNS
# ---------------------------------------------------------

pattern_files = [
    data["dreaddit_pattern_overall"],
    data["dreaddit_pattern_score"],
    data["dreaddit_pattern_threshold"],
    data["dreaddit_pattern_subreddit"],
    data["dreaddit_pattern_text_length"],
]

pattern_records = sum(
    len(x) for x in pattern_files
    if x is not None
)

report += make_section(
    "7. Observed Error Patterns",
    (
        f"Existing Dreaddit error-pattern reports contain "
        f"{pattern_records} combined records across the available "
        f"pattern summaries.\n\n"
        "The available analyses cover overall error behavior, model "
        "score bands, threshold behavior, subreddit-level patterns "
        "and text-length patterns. These are useful for understanding "
        "where the model makes mistakes within the dataset.\n\n"
        "They should not be interpreted as evidence that a particular "
        "subreddit, writing style, or text length causes mental-health "
        "risk."
    ),
)

# ---------------------------------------------------------
# CROSS-MODEL ERROR OBSERVATIONS
# ---------------------------------------------------------

report += make_section(
    "8. Cross-Experiment Error Observations",
    (
        "Several consistent observations emerge:\n\n"
        "1. **Accuracy alone is insufficient.** EATD experiments "
        "illustrate how a model can obtain moderate accuracy while "
        "missing most positive examples.\n\n"
        "2. **Positive-class recall is a major limitation.** "
        "EATD text, voice and multimodal experiments all showed "
        "limited detection of the positive class.\n\n"
        "3. **Multimodal fusion did not automatically improve "
        "discrimination.** Combining text and voice under the "
        "tested EATD protocol produced weak ROC-AUC and PR-AUC.\n\n"
        "4. **Dataset-specific evaluation matters.** Results from "
        "Dreaddit, EATD and StudentLife use different populations, "
        "targets and evaluation protocols and therefore should not "
        "be interpreted as one common benchmark.\n\n"
        "5. **Threshold changes affect operating characteristics "
        "but do not establish clinical validity.**"
    ),
)

# ---------------------------------------------------------
# LIMITATIONS
# ---------------------------------------------------------

report += make_section(
    "9. Error-Analysis Limitations",
    (
        "- EATD has a relatively small participant count and "
        "substantial class imbalance.\n"
        "- EATD text and voice experiments showed weak positive-class "
        "discrimination.\n"
        "- StudentLife behavioral modeling does not constitute true "
        "temporal early prediction because exact assessment-time "
        "alignment was unavailable.\n"
        "- Dreaddit contained three exact cross-split text overlaps, "
        "which are retained as an evaluation limitation.\n"
        "- The datasets differ in population, collection protocol, "
        "label definition and modality.\n"
        "- MODMA raw-audio ML benchmarking was not completed because "
        "controlled audio access was unavailable.\n"
        "- Error patterns are dataset-specific and cannot be assumed "
        "to generalize to clinical populations.\n"
        "- The framework is research-only and does not provide a "
        "clinical diagnosis or clinical risk determination."
    ),
)

# ---------------------------------------------------------
# RESEARCH IMPLICATIONS
# ---------------------------------------------------------

report += make_section(
    "10. Research Implications",
    (
        "The error analysis supports a cautious research conclusion: "
        "different modalities provide experimentally measurable "
        "signals, but the current datasets and baselines do not "
        "justify treating the framework as a clinically validated "
        "early-detection system.\n\n"
        "The strongest experimental evidence in the current project "
        "comes from the Dreaddit text experiments, while the EATD "
        "experiments demonstrate important limitations in positive-"
        "class discrimination. StudentLife provides a behavioral "
        "association baseline but lacks the temporal alignment "
        "required for a true early-warning evaluation.\n\n"
        "These findings justify further work on larger datasets, "
        "better temporal alignment, class-imbalance handling, "
        "external validation, calibration and prospective evaluation "
        "before any consequential use could be considered."
    ),
)

# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

report += make_section(
    "Phase 10.3 Status",
    "COMPLETE",
)

OUTPUT.write_text(
    report,
    encoding="utf-8"
)

print("=" * 60)
print("PHASE 10.3 COMPLETE")
print("=" * 60)
print(f"Created: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size} bytes")
print("=" * 60)