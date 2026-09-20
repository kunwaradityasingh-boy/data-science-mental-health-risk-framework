"""
Data Science Framework for Early Detection of Mental Health Risk
===============================================================

Multimodal Research Dashboard

Live components:
    1. Text    -> Dreaddit TF-IDF + Logistic Regression
    2. Voice   -> EATD acoustic features + Logistic Regression
    3. Behavior-> StudentLife behavioral features + Random Forest Regression

Important:
    Research prototype only.
    Not a clinical diagnostic system.
    Model outputs are not medical probabilities.
    Cross-dataset probability fusion is intentionally disabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# MODEL PATHS
# ============================================================

TEXT_MODEL_DIR = ROOT_DIR / "models" / "text_baseline"
VOICE_MODEL_DIR = ROOT_DIR / "models" / "eatd_voice"
BEHAVIOR_MODEL_DIR = ROOT_DIR / "models" / "studentlife_behavior"

TEXT_VECTORIZER = (
    TEXT_MODEL_DIR / "tfidf_vectorizer.joblib"
)

TEXT_MODEL = (
    TEXT_MODEL_DIR / "logistic_regression.joblib"
)

VOICE_MODEL = (
    VOICE_MODEL_DIR / "logistic_regression.joblib"
)

BEHAVIOR_MODEL = (
    BEHAVIOR_MODEL_DIR / "random_forest_baseline.joblib"
)


# ============================================================
# LOCAL DATA PATHS
# ============================================================

EATD_PROCESSED = (
    ROOT_DIR
    / "data"
    / "processed"
    / "eatd"
    / "acoustic_features.csv"
)

STUDENTLIFE_PROCESSED = (
    ROOT_DIR
    / "data"
    / "processed"
    / "studentlife"
    / "behavior_fused_pre.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mental Health Risk Research Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #F4F7F6;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: white;
    border: 1px solid #D8E1DE;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 22px;
}

.hero h1 {
    margin: 0;
    font-size: 38px;
}

.hero p {
    color: #5B6E6A;
    font-size: 17px;
    max-width: 900px;
}

.card {
    background: white;
    border: 1px solid #D8E1DE;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.status {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    background: #E4F3EA;
    color: #28744A;
}

.warning-box {
    background: #FFF7E6;
    border: 1px solid #E8CF91;
    border-radius: 12px;
    padding: 16px;
}

.danger-box {
    background: #FFF0F0;
    border: 1px solid #E0B0B0;
    border-radius: 12px;
    padding: 16px;
}

.architecture {
    background: #102A26;
    color: white;
    padding: 25px;
    border-radius: 15px;
    font-family: monospace;
    line-height: 1.8;
    overflow-x: auto;
}

.small {
    color: #657570;
    font-size: 14px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_text_models():

    vectorizer = joblib.load(
        TEXT_VECTORIZER
    )

    model = joblib.load(
        TEXT_MODEL
    )

    return vectorizer, model


@st.cache_resource
def load_voice_model():

    return joblib.load(
        VOICE_MODEL
    )


@st.cache_resource
def load_behavior_model():

    return joblib.load(
        BEHAVIOR_MODEL
    )


# ============================================================
# SAFE MODEL AVAILABILITY
# ============================================================

TEXT_AVAILABLE = (
    TEXT_VECTORIZER.exists()
    and TEXT_MODEL.exists()
)

VOICE_AVAILABLE = (
    VOICE_MODEL.exists()
)

BEHAVIOR_AVAILABLE = (
    BEHAVIOR_MODEL.exists()
)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def extract_voice_features(
    audio_bytes: bytes,
) -> pd.DataFrame:

    import tempfile

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp:

        temp.write(audio_bytes)
        temp_path = Path(temp.name)

    try:

        signal, sr = librosa.load(
            temp_path,
            sr=16000,
            mono=True,
        )

        if signal.size == 0:
            raise ValueError(
                "Audio contains zero samples."
            )

        duration = len(signal) / sr

        if duration <= 0:
            raise ValueError(
                "Audio duration is zero."
            )

        rms = librosa.feature.rms(
            y=signal
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            signal
        )[0]

        centroid = librosa.feature.spectral_centroid(
            y=signal,
            sr=sr,
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=signal,
            sr=sr,
        )[0]

        rolloff = librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr,
        )[0]

        contrast = librosa.feature.spectral_contrast(
            y=signal,
            sr=sr,
        )

        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=sr,
        )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13,
        )

        delta_mfcc = librosa.feature.delta(
            mfcc
        )

        delta2_mfcc = librosa.feature.delta(
            mfcc,
            order=2,
        )

        features = {

            "duration_sec": duration,

            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_median": float(np.median(rms)),

            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "zcr_median": float(np.median(zcr)),

            "spectral_centroid_mean":
                float(np.mean(centroid)),

            "spectral_centroid_std":
                float(np.std(centroid)),

            "spectral_bandwidth_mean":
                float(np.mean(bandwidth)),

            "spectral_bandwidth_std":
                float(np.std(bandwidth)),

            "spectral_rolloff_mean":
                float(np.mean(rolloff)),

            "spectral_rolloff_std":
                float(np.std(rolloff)),
        }

        # Spectral contrast

        for i in range(
            contrast.shape[0]
        ):

            features[
                f"spectral_contrast_{i+1}_mean"
            ] = float(
                np.mean(contrast[i])
            )

            features[
                f"spectral_contrast_{i+1}_std"
            ] = float(
                np.std(contrast[i])
            )

        # Chroma

        for i in range(
            chroma.shape[0]
        ):

            features[
                f"chroma_{i+1}_mean"
            ] = float(
                np.mean(chroma[i])
            )

            features[
                f"chroma_{i+1}_std"
            ] = float(
                np.std(chroma[i])
            )

        # MFCC + delta + delta2

        for i in range(
            mfcc.shape[0]
        ):

            coefficient = i + 1

            features[
                f"mfcc_{coefficient}_mean"
            ] = float(
                np.mean(mfcc[i])
            )

            features[
                f"mfcc_{coefficient}_std"
            ] = float(
                np.std(mfcc[i])
            )

            features[
                f"mfcc_{coefficient}_median"
            ] = float(
                np.median(mfcc[i])
            )

            features[
                f"delta_mfcc_{coefficient}_mean"
            ] = float(
                np.mean(delta_mfcc[i])
            )

            features[
                f"delta_mfcc_{coefficient}_std"
            ] = float(
                np.std(delta_mfcc[i])
            )

            features[
                f"delta2_mfcc_{coefficient}_mean"
            ] = float(
                np.mean(delta2_mfcc[i])
            )

            features[
                f"delta2_mfcc_{coefficient}_std"
            ] = float(
                np.std(delta2_mfcc[i])
            )

        return pd.DataFrame([features])

    finally:

        try:
            temp_path.unlink()
        except Exception:
            pass


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(text):

    vectorizer, model = load_text_models()

    X = vectorizer.transform(
        [text]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    prediction = int(
        probability >= 0.50
    )

    return probability, prediction


# ============================================================
# VOICE PREDICTION
# ============================================================

def predict_voice(audio_bytes):

    model = load_voice_model()

    features = extract_voice_features(
        audio_bytes
    )

    expected_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if expected_features is not None:

        missing = [
            column
            for column in expected_features
            if column not in features.columns
        ]

        if missing:

            raise ValueError(
                "Voice feature mismatch. "
                f"Missing {len(missing)} features."
            )

        features = features[
            list(expected_features)
        ]

    probability = float(
        model.predict_proba(
            features
        )[0][1]
    )

    prediction = int(
        probability >= 0.50
    )

    return (
        probability,
        prediction,
        features,
    )


# ============================================================
# BEHAVIOR PREDICTION
# ============================================================

def predict_behavior(
    behavior_df,
):

    model = load_behavior_model()

    expected_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if expected_features is None:

        expected_count = getattr(
            model,
            "n_features_in_",
            None,
        )

        if expected_count is not None:
            expected_features = [
                f"feature_{i+1}"
                for i in range(
                    expected_count
                )
            ]

    if expected_features is not None:

        expected_features = list(
            expected_features
        )

        missing = [
            column
            for column in expected_features
            if column not in behavior_df.columns
        ]

        if missing:

            raise ValueError(
                "Behavior feature mismatch.\n\n"
                f"Missing columns: {missing}"
            )

        behavior_df = behavior_df[
            expected_features
        ].copy()

    behavior_df = behavior_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if behavior_df.isna().any().any():

        raise ValueError(
            "Behavior input contains missing "
            "or non-numeric feature values."
        )

    if np.isinf(
        behavior_df.to_numpy()
    ).any():

        raise ValueError(
            "Behavior input contains infinite values."
        )

    score = float(
        model.predict(
            behavior_df
        )[0]
    )

    return score


# ============================================================
# EXAMPLES
# ============================================================

TEXT_EXAMPLE = (
    "I have been feeling overwhelmed by my studies "
    "and upcoming exams. I find it difficult to "
    "concentrate, and I often worry about my performance. "
    "I feel stressed because of the workload and deadlines."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Research Framework"
    )

    st.caption(
        "Multimodal Research Dashboard v2.0"
    )

    st.divider()

    st.markdown(
        "### Component status"
    )

    if TEXT_AVAILABLE:
        st.success(
            "✅ Text model available"
        )
    else:
        st.error(
            "❌ Text model missing"
        )

    if VOICE_AVAILABLE:
        st.success(
            "✅ Voice model available"
        )
    else:
        st.warning(
            "⚠️ Voice model missing"
        )

    if BEHAVIOR_AVAILABLE:
        st.success(
            "✅ Behavior model available"
        )
    else:
        st.warning(
            "⚠️ Behavior model missing"
        )

    st.divider()

    st.markdown(
        """
        ### Completed phases

        ✅ Phase 2 — Data schemas

        ✅ Phase 3 — Dreaddit text

        ✅ Phase 4 — MODMA research/access

        ✅ Phase 5 — StudentLife behavior

        ✅ Phase 6 — EATD voice

        ✅ Phase 7 — EATD text

        ✅ Phase 8 — Text + Voice fusion

        ✅ Phase 9 — Integration & governance

        ✅ Phase 10 — Research analysis
        """
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<h1>🧠 Data Science Framework for Early Detection of Mental Health Risk</h1>

<p>
A multimodal research framework investigating text,
voice and behavioral signals using dataset-specific
machine-learning pipelines.
</p>

<span class="status">
Research Prototype
</span>

&nbsp;

<span class="status">
Phase 10 Complete
</span>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

(
    text_tab,
    voice_tab,
    behavior_tab,
    results_tab,
    architecture_tab,
    governance_tab,
) = st.tabs(
    [
        "💬 Text",
        "🎙️ Voice",
        "📱 Behavior",
        "📊 Results",
        "🏗️ Architecture",
        "🛡️ Governance",
    ]
)


# ============================================================
# TEXT TAB
# ============================================================

with text_tab:

    st.subheader(
        "💬 Text Research Demo"
    )

    st.write(
        "Dreaddit text baseline using "
        "TF-IDF + Logistic Regression."
    )

    if not TEXT_AVAILABLE:

        st.error(
            "Text model artifacts are not available."
        )

    else:

        text = st.text_area(
            "Enter text",
            value=TEXT_EXAMPLE,
            height=200,
        )

        if st.button(
            "🔍 Analyze Text",
            type="primary",
            use_container_width=True,
        ):

            if len(text.split()) < 8:

                st.warning(
                    "Text is very short. "
                    "Interpretation may be unstable."
                )

            try:

                probability, prediction = (
                    predict_text(text)
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Class-1 model score",
                    f"{probability * 100:.2f}%",
                )

                c2.metric(
                    "Classification",
                    (
                        "Class 1"
                        if prediction
                        else "Class 0"
                    ),
                )

                c3.metric(
                    "Words",
                    len(text.split()),
                )

                if prediction:

                    st.warning(
                        "Stress-related classification "
                        "signal detected by the research model."
                    )

                else:

                    st.success(
                        "No stress-related classification "
                        "signal detected by the research model."
                    )

                st.caption(
                    "This is a Dreaddit research classification "
                    "score, not a clinical probability or diagnosis."
                )

            except Exception as error:

                st.error(
                    "Text prediction failed."
                )

                st.exception(error)


# ============================================================
# VOICE TAB
# ============================================================

with voice_tab:

    st.subheader(
        "🎙️ Voice Research Demo"
    )

    st.write(
        "Upload a WAV recording. The dashboard extracts "
        "the same acoustic feature family used by the "
        "EATD voice baseline and sends it to the trained "
        "Logistic Regression pipeline."
    )

    st.info(
        "EATD voice target: standardized SDS > 52. "
        "The output is a research classification signal, "
        "not a diagnosis."
    )

    if not VOICE_AVAILABLE:

        st.error(
            "Voice model artifact is not available."
        )

        st.code(
            str(VOICE_MODEL)
        )

    else:

        audio_file = st.file_uploader(
            "Upload a WAV file",
            type=["wav"],
            key="voice_upload",
        )

        if audio_file is not None:

            st.audio(
                audio_file.getvalue(),
                format="audio/wav",
            )

            if st.button(
                "🎙️ Analyze Voice",
                type="primary",
                use_container_width=True,
            ):

                try:

                    audio_bytes = (
                        audio_file.getvalue()
                    )

                    probability, prediction, features = (
                        predict_voice(
                            audio_bytes
                        )
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Class-1 model score",
                        f"{probability * 100:.2f}%",
                    )

                    c2.metric(
                        "Classification",
                        (
                            "Class 1"
                            if prediction
                            else "Class 0"
                        ),
                    )

                    c3.metric(
                        "Acoustic features",
                        len(features.columns),
                    )

                    if prediction:

                        st.warning(
                            "EATD Class-1 classification "
                            "signal detected."
                        )

                    else:

                        st.success(
                            "EATD Class-0 classification "
                            "signal detected."
                        )

                    st.subheader(
                        "Extracted acoustic features"
                    )

                    st.dataframe(
                        features.T.rename(
                            columns={0: "value"}
                        ),
                        use_container_width=True,
                    )

                    st.caption(
                        "The feature table is shown for "
                        "research transparency. Acoustic "
                        "features do not constitute clinical evidence."
                    )

                except Exception as error:

                    st.error(
                        "Voice prediction failed."
                    )

                    st.exception(error)


# ============================================================
# BEHAVIOR TAB
# ============================================================

with behavior_tab:

    st.subheader(
        "📱 Behavioral Research Demo"
    )

    st.write(
        "Upload a participant-level StudentLife behavioral "
        "feature CSV. The trained Random Forest regression "
        "model estimates the PHQ-9 symptom score used in "
        "the completed StudentLife baseline experiment."
    )

    st.warning(
        "Important: StudentLife behavioral modeling is a "
        "retrospective participant-level association baseline. "
        "The source PHQ-9 table did not provide an exact "
        "assessment timestamp, so this is NOT presented as "
        "temporally aligned early prediction."
    )

    if not BEHAVIOR_AVAILABLE:

        st.error(
            "Behavior model artifact is not available."
        )

        st.code(
            str(BEHAVIOR_MODEL)
        )

    else:

        behavior_model = load_behavior_model()

        expected_features = getattr(
            behavior_model,
            "feature_names_in_",
            None,
        )

        if expected_features is not None:

            expected_features = list(
                expected_features
            )

            st.write(
                f"Expected behavioral features: "
                f"**{len(expected_features)}**"
            )

            template_df = pd.DataFrame(
                [np.zeros(
                    len(expected_features)
                )],
                columns=expected_features,
            )

            st.download_button(
                "⬇️ Download behavior CSV template",
                data=template_df.to_csv(
                    index=False
                ),
                file_name=(
                    "studentlife_behavior_template.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander(
                "Show expected feature names"
            ):

                st.dataframe(
                    pd.DataFrame(
                        {
                            "feature": expected_features
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        behavior_file = st.file_uploader(
            "Upload behavioral feature CSV",
            type=["csv"],
            key="behavior_upload",
        )

        if behavior_file is not None:

            try:

                uploaded_df = pd.read_csv(
                    behavior_file
                )

                st.write(
                    "Uploaded data preview"
                )

                st.dataframe(
                    uploaded_df.head(),
                    use_container_width=True,
                )

                if st.button(
                    "📱 Analyze Behavior",
                    type="primary",
                    use_container_width=True,
                ):

                    score = predict_behavior(
                        uploaded_df
                    )

                    st.metric(
                        "Estimated PHQ-9 research score",
                        f"{score:.2f}",
                    )

                    st.caption(
                        "This is a model-estimated PHQ-9 "
                        "symptom score from the StudentLife "
                        "behavioral baseline. It is not a "
                        "clinical assessment."
                    )

            except Exception as error:

                st.error(
                    "Behavior prediction failed."
                )

                st.exception(error)


# ============================================================
# RESULTS TAB
# ============================================================

with results_tab:

    st.subheader(
        "📊 Completed Research Results"
    )

    st.write(
        "These results come from separate dataset-specific "
        "experiments. They must not be interpreted as one "
        "common benchmark."
    )

    results = pd.DataFrame(
        [
            [
                "Dreaddit Text",
                "Classification",
                "Final holdout",
                "71.75%",
                "73.63%",
                "82.07%",
                "84.12%",
            ],
            [
                "Dreaddit Text + Engineered",
                "Classification",
                "Post-aware validation",
                "77.66%",
                "78.96%",
                "85.03%",
                "84.78%",
            ],
            [
                "EATD Voice",
                "Classification",
                "Official validation",
                "77.22%",
                "18.18%",
                "55.31%",
                "23.88%",
            ],
            [
                "EATD Text",
                "Classification",
                "Official validation",
                "78.48%",
                "0.00%",
                "50.94%",
                "21.05%",
            ],
            [
                "EATD Text + Voice",
                "Classification",
                "Official validation",
                "74.68%",
                "16.67%",
                "51.67%",
                "21.52%",
            ],
            [
                "StudentLife Behavior",
                "Regression",
                "5-fold participant CV",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            ],
        ],
        columns=[
            "Model",
            "Task",
            "Evaluation",
            "Accuracy",
            "F1",
            "ROC-AUC",
            "PR-AUC",
        ],
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The EATD experiments show weak positive-class "
        "detection despite relatively high accuracy. "
        "Accuracy alone should not be used to claim model success."
    )

    st.subheader(
        "StudentLife behavioral baseline"
    )

    st.write(
        "Random Forest regression:"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        "3.8488",
    )

    c2.metric(
        "RMSE",
        "4.7222",
    )

    c3.metric(
        "R²",
        "-0.1510",
    )

    st.caption(
        "Negative R² indicates that the regression baseline "
        "did not explain the held-out variance well."
    )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with architecture_tab:

    st.subheader(
        "🏗️ Final Multimodal Architecture"
    )

    st.markdown(
        """
<div class="architecture">

Data Sources
      ↓
Consent / Provenance / Governance
      ↓
Data Quality Validation
      ↓
┌────────────────────────────────────────────┐
│                                            │
│  TEXT                                      │
│  Dreaddit                                  │
│  TF-IDF + Logistic Regression              │
│                                            │
│  VOICE                                     │
│  EATD                                      │
│  Acoustic Features + Logistic Regression  │
│                                            │
│  BEHAVIOR                                  │
│  StudentLife                               │
│  Behavioral Features + RF Regression       │
│                                            │
└────────────────────────────────────────────┘
      ↓
Dataset-Specific Evaluation
      ↓
Error Analysis
      ↓
Model Registry
      ↓
Unified Prediction Schema
      ↓
Risk Policy
      ↓
Framework Prediction Service
      ↓
Human Review
      ↓
Research Output

</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    architecture_df = pd.DataFrame(
        [
            [
                "Text",
                "Dreaddit",
                "TF-IDF + Logistic Regression",
                "Live",
            ],
            [
                "Voice",
                "EATD",
                "142 acoustic features + Logistic Regression",
                "Live",
            ],
            [
                "Behavior",
                "StudentLife",
                "RF regression → PHQ-9",
                "Live",
            ],
            [
                "Text + Voice",
                "EATD",
                "OOF multimodal fusion",
                "Research",
            ],
            [
                "Integration",
                "Framework",
                "Unified prediction schema",
                "Complete",
            ],
            [
                "Governance",
                "Framework",
                "Risk policy + human review",
                "Complete",
            ],
            [
                "API",
                "Framework",
                "FastAPI",
                "Complete",
            ],
            [
                "Dashboard",
                "Framework",
                "Streamlit",
                "Live",
            ],
        ],
        columns=[
            "Component",
            "Dataset",
            "Method",
            "Status",
        ],
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Cross-dataset probability fusion is intentionally "
        "disabled because Dreaddit, EATD and StudentLife use "
        "different populations, targets and evaluation protocols."
    )


# ============================================================
# GOVERNANCE TAB
# ============================================================

with governance_tab:

    st.subheader(
        "🛡️ Governance & Responsible Research"
    )

    st.markdown(
        """
<div class="danger-box">

<strong>Research-only system</strong>

<br><br>

This framework does not diagnose mental-health disorders,
does not replace qualified professionals and must not be
used as an automated consequential decision system.

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 🚫 Prohibited use"
    )

    st.markdown(
        """
        - Clinical diagnosis
        - Emergency triage
        - Automated treatment decisions
        - Academic punishment
        - Employment decisions
        - Insurance decisions
        - Legal decisions
        - Treating model scores as medical facts
        """
    )

    st.markdown(
        "### 🔐 Privacy"
    )

    st.markdown(
        """
        - Pseudonymous identifiers
        - Consent-aware data handling
        - Data minimization
        - Restricted raw datasets
        - No raw sensitive dataset upload through the dashboard
        - Secure secrets management
        - Model and input validation
        """
    )

    st.markdown(
        "### 👤 Human-in-the-loop"
    )

    st.info(
        "Any consequential interpretation requires appropriately "
        "authorized human review. Model output is a research signal, "
        "not an autonomous decision."
    )

    st.markdown(
        "### 📌 Interpretation"
    )

    st.write(
        "Text and voice outputs are classification model scores. "
        "The StudentLife output is a regression estimate of the "
        "PHQ-9 research target. None of these outputs should be "
        "interpreted as a person's clinical diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div style="text-align:center;color:#657570;font-size:13px;">

🧠 Data Science Framework for Early Detection of Mental Health Risk

<br>

Multimodal Research Prototype • Phase 10 Complete

<br>

Text • Voice • Behavior • Integration • Governance

<br><br>

Not a diagnostic or clinical decision-making system.

</div>
""",
    unsafe_allow_html=True,
)