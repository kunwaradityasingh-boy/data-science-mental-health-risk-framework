"""
Data Science Framework for Early Detection of Mental Health Risk
===============================================================

Polished Multimodal Research Dashboard

UI-only redesign:
    - Existing model paths are preserved.
    - Existing prediction functions are preserved.
    - Existing dataset-specific methodology is preserved.
    - No cross-dataset probability fusion is introduced.

Research prototype only. Not a clinical diagnostic system.
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

TEXT_VECTORIZER = TEXT_MODEL_DIR / "tfidf_vectorizer.joblib"
TEXT_MODEL = TEXT_MODEL_DIR / "logistic_regression.joblib"
VOICE_MODEL = VOICE_MODEL_DIR / "logistic_regression.joblib"
BEHAVIOR_MODEL = BEHAVIOR_MODEL_DIR / "random_forest_baseline.joblib"


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
# POLISHED UI CSS
# ============================================================

st.markdown(
    """
<style>
/* ---------- Global ---------- */
.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(20, 184, 166, 0.08), transparent 28%),
        radial-gradient(circle at 0% 20%, rgba(59, 130, 246, 0.06), transparent 25%),
        #f6f8fa;
    color: #102a2a;
}

.block-container {
    max-width: 1380px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2928 0%, #123936 55%, #0b2524 100%);
}

section[data-testid="stSidebar"] * {
    color: #ecfdf8 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.14);
}

.sidebar-brand {
    padding: 8px 2px 18px 2px;
}

.sidebar-brand .brain {
    font-size: 34px;
}

.sidebar-brand h2 {
    margin: 4px 0 2px 0;
    font-size: 21px;
    color: white;
}

.sidebar-brand p {
    margin: 0;
    font-size: 12px;
    color: #a8c7c1 !important;
}

.sidebar-section {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #86b8ae !important;
    font-weight: 700;
    margin: 18px 0 9px 0;
}

.side-status {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 11px;
    margin: 7px 0;
    border-radius: 11px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 13px;
}

.side-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    flex: 0 0 auto;
}

.dot-green { background: #34d399; box-shadow: 0 0 10px rgba(52,211,153,.55); }
.dot-yellow { background: #fbbf24; box-shadow: 0 0 10px rgba(251,191,36,.45); }
.dot-red { background: #f87171; }

.phase-item {
    padding: 6px 0;
    color: #c8dfda !important;
    font-size: 13px;
}

/* ---------- Hero ---------- */
.hero-shell {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0d2928 0%, #164e49 62%, #0f766e 100%);
    border-radius: 24px;
    padding: 34px 38px;
    margin-bottom: 20px;
    box-shadow: 0 18px 50px rgba(13,41,40,.16);
}

.hero-shell:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -65px;
    top: -75px;
    border-radius: 50%;
    border: 35px solid rgba(255,255,255,.07);
}

.hero-kicker {
    color: #8de5d5;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    color: white;
    font-size: clamp(30px, 4vw, 48px);
    line-height: 1.08;
    font-weight: 800;
    margin: 0;
    max-width: 900px;
}

.hero-subtitle {
    color: #cce9e4;
    font-size: 16px;
    line-height: 1.6;
    max-width: 880px;
    margin: 14px 0 18px 0;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    margin-right: 7px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 750;
}

.badge-light {
    color: #e9fffa;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
}

.badge-green {
    color: #063b30;
    background: #b7f4df;
}

/* ---------- Top stats ---------- */
.stat-card {
    background: rgba(255,255,255,.88);
    border: 1px solid #e2e8e7;
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 96px;
    box-shadow: 0 5px 20px rgba(16,42,42,.045);
}

.stat-label {
    color: #6b7f7c;
    font-size: 12px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.stat-value {
    color: #102a2a;
    font-size: 24px;
    font-weight: 800;
    margin-top: 5px;
}

.stat-note {
    color: #78908b;
    font-size: 11px;
    margin-top: 2px;
}

/* ---------- Section ---------- */
.section-title {
    color: #102a2a;
    font-size: 25px;
    font-weight: 800;
    margin: 4px 0 5px 0;
}

.section-subtitle {
    color: #647773;
    font-size: 14px;
    margin-bottom: 17px;
}

.info-card {
    background: white;
    border: 1px solid #e1e8e6;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 7px 25px rgba(16,42,42,.045);
}

.info-card h4 {
    margin: 0 0 7px 0;
    color: #173c39;
}

.info-card p {
    color: #637570;
    line-height: 1.55;
    margin-bottom: 0;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 11px !important;
    font-weight: 750 !important;
    min-height: 45px;
    border: 1px solid #d8e3e0 !important;
    transition: all .15s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    border: 0 !important;
    color: white !important;
    box-shadow: 0 7px 18px rgba(13,148,136,.18);
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* ---------- Inputs ---------- */
.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stFileUploader {
    border-radius: 11px !important;
}

/* ---------- Result cards ---------- */
.result-card {
    background: white;
    border: 1px solid #e1e8e6;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 8px 28px rgba(16,42,42,.05);
}

.result-label {
    color: #71817e;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 800;
}

.result-value {
    color: #102a2a;
    font-size: 28px;
    font-weight: 850;
    margin-top: 6px;
}

.result-muted {
    color: #71817e;
    font-size: 12px;
    margin-top: 3px;
}

/* ---------- Status pills ---------- */
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #e7f8f1;
    color: #16745a;
    font-size: 12px;
    font-weight: 800;
}

.live-pill span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
}

/* ---------- Architecture ---------- */
.architecture {
    background: linear-gradient(135deg, #0b2524, #123936);
    color: #eafff9;
    padding: 28px;
    border-radius: 18px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    line-height: 1.85;
    overflow-x: auto;
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: 0 14px 35px rgba(8,34,32,.16);
}

/* ---------- Governance ---------- */
.gov-card {
    background: white;
    border: 1px solid #e1e8e6;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 13px;
}

.gov-card h4 {
    color: #173c39;
    margin: 0 0 8px 0;
}

.gov-card p, .gov-card li {
    color: #637570;
    line-height: 1.55;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #7b8d89;
    font-size: 12px;
    padding: 22px 0 5px 0;
}

.footer strong {
    color: #365954;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* ============================================================
   INDUSTRY / ENTERPRISE UI LAYER
   ============================================================ */

:root {
    --ink: #102a2a;
    --muted: #647773;
    --line: #dfe8e5;
    --surface: #ffffff;
    --surface-soft: #f8fbfa;
    --brand: #0f766e;
    --brand-dark: #0b3b38;
}

.main .block-container {
    max-width: 1440px;
}

/* Enterprise top utility bar */
.enterprise-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 14px;
    margin-bottom: 12px;
    border: 1px solid #dce8e4;
    border-radius: 10px;
    background: rgba(255,255,255,.86);
    box-shadow: 0 4px 16px rgba(16,42,42,.035);
    font-size: 12px;
}

.enterprise-left {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #42615b;
    font-weight: 700;
}

.enterprise-right {
    color: #73847f;
}

.enterprise-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 4px rgba(34,197,94,.10);
}

/* Better hero */
.hero-shell {
    min-height: 285px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-radius: 22px;
    padding: 40px 44px;
    background:
        radial-gradient(circle at 88% 28%, rgba(126,245,220,.16), transparent 18%),
        radial-gradient(circle at 75% 105%, rgba(59,130,246,.12), transparent 28%),
        linear-gradient(120deg, #092d2b 0%, #0e4641 52%, #0f766e 100%);
    box-shadow: 0 20px 55px rgba(9,45,43,.18);
}

.hero-kicker {
    font-size: 11px;
    letter-spacing: .16em;
}

.hero-title {
    max-width: 1000px;
    font-size: clamp(32px, 4.1vw, 54px);
    letter-spacing: -.025em;
}

.hero-subtitle {
    max-width: 940px;
    font-size: 16px;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 2px;
}

.meta-chip {
    padding: 7px 10px;
    border-radius: 8px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.13);
    color: #d9efeb;
    font-size: 11px;
    font-weight: 700;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(180deg, #ffffff, #fbfdfc);
    border: 1px solid #dfe8e5;
    border-radius: 14px;
    padding: 17px 18px;
    min-height: 104px;
    box-shadow: 0 6px 20px rgba(16,42,42,.04);
}

.kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.kpi-icon {
    font-size: 19px;
}

.kpi-label {
    color: #71817e;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .09em;
    text-transform: uppercase;
}

.kpi-value {
    color: #102a2a;
    font-size: 27px;
    line-height: 1.1;
    font-weight: 850;
    margin-top: 8px;
}

.kpi-note {
    color: #80918d;
    font-size: 11px;
    margin-top: 4px;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 750 !important;
    color: #647773 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0f766e !important;
}

/* Better dataframe container */
[data-testid="stDataFrame"] {
    border: 1px solid #dfe8e5;
    border-radius: 12px;
    overflow: hidden;
}

/* Section separators */
.section-rule {
    height: 1px;
    background: linear-gradient(90deg, #dfe8e5, transparent);
    margin: 8px 0 20px 0;
}

/* Model cards */
.model-card {
    background: #fff;
    border: 1px solid #dfe8e5;
    border-radius: 14px;
    padding: 16px;
    min-height: 130px;
    box-shadow: 0 5px 18px rgba(16,42,42,.035);
}

.model-card .model-name {
    color: #173c39;
    font-size: 15px;
    font-weight: 800;
}

.model-card .model-meta {
    color: #748681;
    font-size: 12px;
    line-height: 1.55;
    margin-top: 8px;
}

.model-live {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 8px;
    border-radius: 6px;
    background: #e8f8f2;
    color: #17765e;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .07em;
}

/* Research note */
.research-note {
    border-left: 3px solid #0f766e;
    background: #f0f9f7;
    border-radius: 0 10px 10px 0;
    padding: 12px 15px;
    color: #506964;
    font-size: 12px;
    line-height: 1.55;
}

/* Footer */
.footer {
    border-top: 1px solid #dfe8e5;
    margin-top: 28px;
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
    vectorizer = joblib.load(TEXT_VECTORIZER)
    model = joblib.load(TEXT_MODEL)
    return vectorizer, model


@st.cache_resource
def load_voice_model():
    return joblib.load(VOICE_MODEL)


@st.cache_resource
def load_behavior_model():
    return joblib.load(BEHAVIOR_MODEL)


# ============================================================
# MODEL AVAILABILITY
# ============================================================

TEXT_AVAILABLE = TEXT_VECTORIZER.exists() and TEXT_MODEL.exists()
VOICE_AVAILABLE = VOICE_MODEL.exists()
BEHAVIOR_AVAILABLE = BEHAVIOR_MODEL.exists()


# ============================================================
# VOICE FEATURE EXTRACTION
# ============================================================

def extract_voice_features(audio_bytes: bytes) -> pd.DataFrame:
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
            raise ValueError("Audio contains zero samples.")

        duration = len(signal) / sr

        if duration <= 0:
            raise ValueError("Audio duration is zero.")

        rms = librosa.feature.rms(y=signal)[0]
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        features = {
            "duration_sec": duration,
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_median": float(np.median(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "zcr_median": float(np.median(zcr)),
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
        }

        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_{i+1}_mean"] = float(np.mean(contrast[i]))
            features[f"spectral_contrast_{i+1}_std"] = float(np.std(contrast[i]))

        for i in range(chroma.shape[0]):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        for i in range(mfcc.shape[0]):
            coefficient = i + 1
            features[f"mfcc_{coefficient}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{coefficient}_std"] = float(np.std(mfcc[i]))
            features[f"mfcc_{coefficient}_median"] = float(np.median(mfcc[i]))
            features[f"delta_mfcc_{coefficient}_mean"] = float(np.mean(delta_mfcc[i]))
            features[f"delta_mfcc_{coefficient}_std"] = float(np.std(delta_mfcc[i]))
            features[f"delta2_mfcc_{coefficient}_mean"] = float(np.mean(delta2_mfcc[i]))
            features[f"delta2_mfcc_{coefficient}_std"] = float(np.std(delta2_mfcc[i]))

        return pd.DataFrame([features])

    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


# ============================================================
# PREDICTION FUNCTIONS — KEPT INTACT
# ============================================================

def predict_text(text):
    vectorizer, model = load_text_models()
    X = vectorizer.transform([text])
    probability = float(model.predict_proba(X)[0][1])
    prediction = int(probability >= 0.50)
    return probability, prediction


def predict_voice(audio_bytes):
    model = load_voice_model()

    features = extract_voice_features(audio_bytes)

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

        features = features[list(expected_features)]

    probability = float(model.predict_proba(features)[0][1])
    prediction = int(probability >= 0.50)

    return probability, prediction, features


def predict_behavior(behavior_df):
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
                for i in range(expected_count)
            ]

    if expected_features is not None:
        expected_features = list(expected_features)

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

        behavior_df = behavior_df[expected_features].copy()

    behavior_df = behavior_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if behavior_df.isna().any().any():
        raise ValueError(
            "Behavior input contains missing "
            "or non-numeric feature values."
        )

    if np.isinf(behavior_df.to_numpy()).any():
        raise ValueError(
            "Behavior input contains infinite values."
        )

    score = float(model.predict(behavior_df)[0])

    return score


# ============================================================
# EXAMPLE
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
        """
        <div class="sidebar-brand">
            <div class="brain">🧠</div>
            <h2>Research Framework</h2>
            <p>Multimodal Research Dashboard · v2.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">System status</div>', unsafe_allow_html=True)

    statuses = [
        ("Text model", TEXT_AVAILABLE),
        ("Voice model", VOICE_AVAILABLE),
        ("Behavior model", BEHAVIOR_AVAILABLE),
    ]

    for label, available in statuses:
        dot = "dot-green" if available else "dot-yellow"
        text = "Available" if available else "Missing"
        st.markdown(
            f"""
            <div class="side-status">
                <span class="side-dot {dot}"></span>
                <span><strong>{label}</strong><br>
                <small>{text}</small></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section">Research progress</div>', unsafe_allow_html=True)

    phases = [
        "Phase 2 — Data schemas",
        "Phase 3 — Dreaddit text",
        "Phase 4 — MODMA research/access",
        "Phase 5 — StudentLife behavior",
        "Phase 6 — EATD voice",
        "Phase 7 — EATD text",
        "Phase 8 — Text + Voice fusion",
        "Phase 9 — Integration & governance",
        "Phase 10 — Research analysis",
    ]

    for phase in phases:
        st.markdown(
            f'<div class="phase-item">✓ {phase}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.caption(
        "Research prototype only. Model outputs are not medical "
        "probabilities or diagnoses."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="enterprise-bar">
        <div class="enterprise-left">
            <span class="enterprise-dot"></span>
            Research environment operational
        </div>
        <div class="enterprise-right">
            Framework v0.1.0 · Dataset-specific evaluation · Clinical use disabled
        </div>
    </div>

    <div class="hero-shell">
        <div class="hero-kicker">DATA SCIENCE · AI RESEARCH · RESPONSIBLE AI</div>
        <h1 class="hero-title">
            Mental Health Risk Research Framework
        </h1>
        <p class="hero-subtitle">
            A production-style research interface for investigating
            text, voice and behavioral signals with reproducible,
            dataset-specific machine-learning pipelines.
        </p>

        <div class="hero-meta">
            <span class="badge badge-green">✓ Phase 10 Complete</span>
            <span class="badge badge-light">🧪 Research Prototype</span>
            <span class="meta-chip">3 Modalities</span>
            <span class="meta-chip">Model Registry</span>
            <span class="meta-chip">Governance Layer</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP OVERVIEW CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

overview = [
    ("🧩", "3", "Modalities", "Text · Voice · Behavior"),
    ("🗂️", "4", "Research datasets", "Dreaddit · EATD · StudentLife · MODMA"),
    ("✓", "10", "Completed phases", "End-to-end research pipeline"),
    ("🛡️", "ON", "Clinical use", "Diagnosis & consequential use disabled"),
]

for column, (icon, value, label, note) in zip([c1, c2, c3, c4], overview):
    with column:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-icon">{icon}</div>
                </div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



st.write("")

st.markdown(
    """
    <div class="section-title">Model portfolio</div>
    <div class="section-subtitle">
        Live research components currently exposed through this interface
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)

portfolio_cards = [
    (
        m1,
        "💬",
        "Text Intelligence",
        "Dreaddit · TF-IDF + Logistic Regression",
        "Classification research signal",
        TEXT_AVAILABLE,
    ),
    (
        m2,
        "🎙️",
        "Voice Analytics",
        "EATD · 142 acoustic features + Logistic Regression",
        "Acoustic classification research signal",
        VOICE_AVAILABLE,
    ),
    (
        m3,
        "📱",
        "Behavior Analytics",
        "StudentLife · Random Forest regression",
        "PHQ-9 research target estimate",
        BEHAVIOR_AVAILABLE,
    ),
]

for col, icon, name, meta, purpose, available in portfolio_cards:
    with col:
        state = "LIVE" if available else "UNAVAILABLE"
        state_class = "model-live" if available else "model-live"
        st.markdown(
            f"""
            <div class="model-card">
                <div class="kpi-icon">{icon}</div>
                <div class="model-name">{name}</div>
                <div class="model-meta">
                    {meta}<br>
                    {purpose}
                </div>
                <span class="{state_class}">{state}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    """
    <div class="research-note">
        <strong>Research boundary:</strong>
        each modality is evaluated within its own dataset and protocol.
        Cross-dataset probability fusion remains intentionally disabled.
        Dashboard outputs are research signals, not medical diagnoses.
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
        "💬  Text",
        "🎙️  Voice",
        "📱  Behavior",
        "📊  Results",
        "🏗️  Architecture",
        "🛡️  Governance",
    ]
)


# ============================================================
# TEXT TAB
# ============================================================

with text_tab:
    st.markdown('<div class="section-title">Text research demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Dreaddit · TF-IDF + Logistic Regression</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        text = st.text_area(
            "Research text",
            value=TEXT_EXAMPLE,
            height=210,
            label_visibility="collapsed",
            placeholder="Enter research text here...",
        )

        if st.button(
            "🔍  Analyze text signal",
            type="primary",
            use_container_width=True,
        ):
            if len(text.split()) < 8:
                st.warning(
                    "Text is very short. Interpretation may be unstable."
                )

            try:
                probability, prediction = predict_text(text)

                st.session_state["text_result"] = {
                    "probability": probability,
                    "prediction": prediction,
                    "words": len(text.split()),
                }

            except Exception as error:
                st.error("Text prediction failed.")
                st.exception(error)

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h4>About this experiment</h4>
                <p>
                    The live demo uses the trained Dreaddit text baseline.
                    The displayed score is a model output for the research
                    classification task and should not be interpreted as
                    clinical probability.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="info-card">
                <h4>Model</h4>
                <p><strong>TF-IDF</strong> feature representation<br>
                <strong>Logistic Regression</strong> classifier<br>
                <strong>Dreaddit</strong> research dataset</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "text_result" in st.session_state:
        result = st.session_state["text_result"]

        st.write("")
        st.markdown("### Research output")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Class-1 model score",
            f"{result['probability'] * 100:.2f}%",
        )

        r2.metric(
            "Classification",
            "Class 1" if result["prediction"] else "Class 0",
        )

        r3.metric(
            "Input words",
            result["words"],
        )

        if result["prediction"]:
            st.warning(
                "Stress-related classification signal detected by the research model."
            )
        else:
            st.success(
                "No stress-related classification signal detected by the research model."
            )

        st.caption(
            "Research interpretation only · not a diagnosis · not a clinical probability."
        )


# ============================================================
# VOICE TAB
# ============================================================

with voice_tab:
    st.markdown('<div class="section-title">Voice research demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">EATD · acoustic features + Logistic Regression</div>',
        unsafe_allow_html=True,
    )

    if not VOICE_AVAILABLE:
        st.error("Voice model artifact is not available.")
        st.code(str(VOICE_MODEL))
    else:
        left, right = st.columns([1.3, 1], gap="large")

        with left:
            audio_file = st.file_uploader(
                "Upload a WAV recording",
                type=["wav"],
                key="voice_upload",
            )

            if audio_file is not None:
                audio_bytes = audio_file.getvalue()
                st.audio(audio_bytes, format="audio/wav")

                if st.button(
                    "🎙️  Analyze voice signal",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        probability, prediction, features = predict_voice(audio_bytes)

                        st.session_state["voice_result"] = {
                            "probability": probability,
                            "prediction": prediction,
                            "features": features,
                        }

                    except Exception as error:
                        st.error("Voice prediction failed.")
                        st.exception(error)

        with right:
            st.markdown(
                """
                <div class="info-card">
                    <h4>Research target</h4>
                    <p>
                        EATD voice classification using acoustic features
                        and Logistic Regression. The output is a research
                        classification signal, not a diagnosis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "voice_result" in st.session_state:
            result = st.session_state["voice_result"]

            st.write("")
            st.markdown("### Research output")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Class-1 model score",
                f"{result['probability'] * 100:.2f}%",
            )

            r2.metric(
                "Classification",
                "Class 1" if result["prediction"] else "Class 0",
            )

            r3.metric(
                "Acoustic features",
                len(result["features"].columns),
            )

            if result["prediction"]:
                st.warning("EATD Class-1 classification signal detected.")
            else:
                st.success("EATD Class-0 classification signal detected.")

            with st.expander("🔬 View extracted acoustic features"):
                st.dataframe(
                    result["features"].T.rename(columns={0: "value"}),
                    use_container_width=True,
                )

            st.caption(
                "Feature values are shown for research transparency."
            )


# ============================================================
# BEHAVIOR TAB
# ============================================================

with behavior_tab:
    st.markdown('<div class="section-title">Behavioral research demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">StudentLife · behavioral features + Random Forest regression</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "StudentLife modeling is a retrospective participant-level association "
        "baseline. It is not presented as temporally aligned early prediction."
    )

    if not BEHAVIOR_AVAILABLE:
        st.error("Behavior model artifact is not available.")
        st.code(str(BEHAVIOR_MODEL))
    else:
        behavior_model = load_behavior_model()

        expected_features = getattr(
            behavior_model,
            "feature_names_in_",
            None,
        )

        if expected_features is not None:
            expected_features = list(expected_features)

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h4>Expected input</h4>
                        <p>
                            The model expects <strong>{len(expected_features)}</strong>
                            behavioral features in participant-level CSV format.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                template_df = pd.DataFrame(
                    [np.zeros(len(expected_features))],
                    columns=expected_features,
                )

                st.download_button(
                    "⬇️  Download CSV template",
                    data=template_df.to_csv(index=False),
                    file_name="studentlife_behavior_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.expander("🔬 Show expected feature names"):
                st.dataframe(
                    pd.DataFrame({"feature": expected_features}),
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
                uploaded_df = pd.read_csv(behavior_file)

                st.markdown("### Input preview")
                st.dataframe(
                    uploaded_df.head(),
                    use_container_width=True,
                    hide_index=True,
                )

                if st.button(
                    "📱  Analyze behavioral signal",
                    type="primary",
                    use_container_width=True,
                ):
                    score = predict_behavior(uploaded_df)
                    st.session_state["behavior_score"] = score

            except Exception as error:
                st.error("Behavior prediction failed.")
                st.exception(error)

        if "behavior_score" in st.session_state:
            st.write("")
            st.markdown("### Research output")

            st.metric(
                "Estimated PHQ-9 research score",
                f"{st.session_state['behavior_score']:.2f}",
            )

            st.caption(
                "Model-estimated StudentLife PHQ-9 research target. "
                "Not a clinical assessment."
            )


# ============================================================
# RESULTS TAB
# ============================================================

with results_tab:
    st.markdown('<div class="section-title">Research performance console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Verified experiment outputs · descriptive comparison only · no cross-dataset ranking</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '''
        <div class="research-note">
            Metrics below belong to different datasets, populations, labels and
            evaluation protocols. They are presented for research traceability,
            not as a common leaderboard.
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.write("")

    results = pd.DataFrame(
        [
            ["Dreaddit Text", "Classification", "Final holdout", "71.75%", "73.63%", "82.07%", "84.12%"],
            ["Dreaddit Text + Engineered", "Classification", "Post-aware validation", "77.66%", "78.96%", "85.03%", "84.78%"],
            ["EATD Voice", "Classification", "Official validation", "77.22%", "18.18%", "55.31%", "23.88%"],
            ["EATD Text", "Classification", "Official validation", "78.48%", "0.00%", "50.94%", "21.05%"],
            ["EATD Text + Voice", "Classification", "Official validation", "74.68%", "16.67%", "51.67%", "21.52%"],
            ["StudentLife Behavior", "Regression", "5-fold participant CV", "N/A", "N/A", "N/A", "N/A"],
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
        "EATD experiments show weak positive-class detection despite relatively "
        "high accuracy. Accuracy alone should not be used to claim model success."
    )

    st.markdown("### StudentLife behavioral baseline")

    a, b, c = st.columns(3)

    a.metric("MAE", "3.8488")
    b.metric("RMSE", "4.7222")
    c.metric("R²", "-0.1510")

    st.caption(
        "Negative R² indicates that the regression baseline did not explain "
        "held-out variance well."
    )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with architecture_tab:
    st.markdown('<div class="section-title">Final multimodal architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">From provenance and quality control to research output</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="architecture">

DATA SOURCES
      ↓
CONSENT · PROVENANCE · GOVERNANCE
      ↓
DATA QUALITY VALIDATION
      ↓
┌──────────────────────────────────────────────────┐
│                                                  │
│  💬 TEXT        Dreaddit                         │
│                  TF-IDF + Logistic Regression    │
│                                                  │
│  🎙️ VOICE       EATD                            │
│                  Acoustic Features + LR          │
│                                                  │
│  📱 BEHAVIOR    StudentLife                     │
│                  Behavioral Features + RF        │
│                                                  │
└──────────────────────────────────────────────────┘
      ↓
DATASET-SPECIFIC EVALUATION
      ↓
ERROR ANALYSIS
      ↓
MODEL REGISTRY
      ↓
UNIFIED PREDICTION SCHEMA
      ↓
RISK POLICY
      ↓
FRAMEWORK PREDICTION SERVICE
      ↓
HUMAN REVIEW
      ↓
RESEARCH OUTPUT

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    architecture_df = pd.DataFrame(
        [
            ["Text", "Dreaddit", "TF-IDF + Logistic Regression", "Live"],
            ["Voice", "EATD", "142 acoustic features + Logistic Regression", "Live"],
            ["Behavior", "StudentLife", "RF regression → PHQ-9", "Live"],
            ["Text + Voice", "EATD", "OOF multimodal fusion", "Research"],
            ["Integration", "Framework", "Unified prediction schema", "Complete"],
            ["Governance", "Framework", "Risk policy + human review", "Complete"],
            ["API", "Framework", "FastAPI", "Complete"],
            ["Dashboard", "Framework", "Streamlit", "Live"],
        ],
        columns=["Component", "Dataset", "Method", "Status"],
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Cross-dataset probability fusion is intentionally disabled because "
        "Dreaddit, EATD and StudentLife use different populations, targets "
        "and evaluation protocols."
    )


# ============================================================
# GOVERNANCE TAB
# ============================================================

with governance_tab:
    st.markdown('<div class="section-title">Governance & responsible research</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Safety boundaries are part of the system architecture</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="gov-card">
            <h4>🛡️ Research-only system</h4>
            <p>
                This framework does not diagnose mental-health disorders,
                does not replace qualified professionals and must not be
                used as an automated consequential decision system.
            </p>
        </div>

        <div class="gov-card">
            <h4>🚫 Prohibited use</h4>
            <ul>
                <li>Clinical diagnosis</li>
                <li>Emergency triage</li>
                <li>Automated treatment decisions</li>
                <li>Academic punishment</li>
                <li>Employment decisions</li>
                <li>Insurance decisions</li>
                <li>Legal decisions</li>
                <li>Treating model scores as medical facts</li>
            </ul>
        </div>

        <div class="gov-card">
            <h4>🔐 Privacy</h4>
            <ul>
                <li>Pseudonymous identifiers</li>
                <li>Consent-aware data handling</li>
                <li>Data minimization</li>
                <li>Restricted raw datasets</li>
                <li>No raw sensitive dataset upload through the dashboard</li>
                <li>Secure secrets management</li>
                <li>Model and input validation</li>
            </ul>
        </div>

        <div class="gov-card">
            <h4>👤 Human-in-the-loop</h4>
            <p>
                Any consequential interpretation requires appropriately
                authorized human review. Model output is a research signal,
                not an autonomous decision.
            </p>
        </div>

        <div class="gov-card">
            <h4>📌 Interpretation</h4>
            <p>
                Text and voice outputs are classification model scores.
                The StudentLife output is a regression estimate of the
                PHQ-9 research target. None of these outputs should be
                interpreted as a person's clinical diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>🧠 Data Science Framework for Early Detection of Mental Health Risk</strong>
        <br>
        Multimodal Research Prototype · Phase 10 Complete
        <br>
        Text · Voice · Behavior · Integration · Governance
        <br><br>
        Not a diagnostic or clinical decision-making system.
    </div>
    """,
    unsafe_allow_html=True,
)
