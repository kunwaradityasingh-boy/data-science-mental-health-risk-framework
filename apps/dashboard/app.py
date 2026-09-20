# import sys
# from pathlib import Path

# import streamlit as st


# # Add project root to Python import path
# PROJECT_ROOT = Path(__file__).resolve().parents[2]

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))


# from src.models.predict_text import predict_text


# st.set_page_config(
#     page_title="Mental Health Risk Research",
#     page_icon="🧠",
#     layout="centered",
# )


# st.title("🧠 Mental Health Risk Research Prototype")

# st.caption(
#     "Privacy-aware text-based research screening prototype"
# )


# st.warning(
#     "Research prototype only. "
#     "This system does not diagnose depression, anxiety, "
#     "or any other mental-health disorder."
# )


# st.markdown("### Text Analysis")

# text = st.text_area(
#     "Enter a text sample for research classification:",
#     height=180,
#     placeholder=(
#         "Example: I have been feeling overwhelmed "
#         "and struggling with my daily responsibilities."
#     ),
# )


# analyze = st.button(
#     "🔍 Analyze Text",
#     type="primary",
# )


# if analyze:

#     if not text.strip():
#         st.error("Please enter some text before analysis.")

#     else:

#         with st.spinner("Analyzing text..."):

#             try:
#                 result = predict_text(text)

#                 prediction = result["prediction"]
#                 probability = result["probability_class_1"]
#                 risk_signal = result["risk_signal"]

#                 st.markdown("---")

#                 st.subheader(
#                     "Research Classification Result"
#                 )

#                 col1, col2 = st.columns(2)

#                 with col1:
#                     st.metric(
#                         "Class-1 Score",
#                         f"{probability:.2%}",
#                     )

#                 with col2:

#                     if prediction == 1:
#                         st.metric(
#                             "Classification",
#                             "Class 1",
#                         )
#                     else:
#                         st.metric(
#                             "Classification",
#                             "Class 0",
#                         )

#                 st.markdown("### Signal")

#                 if prediction == 1:
#                     st.warning(
#                         f"⚠️ {risk_signal}"
#                     )
#                 else:
#                     st.info(
#                         f"ℹ️ {risk_signal}"
#                     )

#                 st.markdown("### Model Information")

#                 st.write(
#                     "Model: TF-IDF + Logistic Regression"
#                 )

#                 st.write(
#                     "Dataset: Dreaddit"
#                 )

#                 st.write(
#                     "Validation strategy: Post-aware split"
#                 )

#                 st.write(
#                     "Validation Accuracy: 75.26%"
#                 )

#                 st.write(
#                     "Validation F1 Score: 77.43%"
#                 )

#                 st.markdown("### Important Limitation")

#                 st.info(
#                     "The output represents a research classification "
#                     "signal learned from the Dreaddit stress dataset. "
#                     "It is not a medical diagnosis, clinical probability, "
#                     "or substitute for professional assessment."
#                 )

#             except Exception as exc:

#                 st.error(
#                     f"Prediction failed: {exc}"
#                 )
"""
Mental Health Risk Research Framework - Streamlit front end (v0.2.0)

Research prototype: TF-IDF + Logistic Regression trained on Dreaddit.
The output is a research classification signal, NOT a diagnosis.
"""

from __future__ import annotations

import math
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = ROOT_DIR / "models" / "text_baseline"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"

APP_VERSION = "0.2.0"

VALIDATION_METRICS = {
    "Validation accuracy": "75.26%",
    "Validation F1 score": "77.43%",
    "Validation ROC-AUC": "84.32%",
}

EXAMPLES = {
    "academic": (
        "Academic pressure",
        "I have been struggling with my studies lately. My assignments are "
        "piling up and I am worried about my upcoming exams. I find it "
        "difficult to concentrate and I keep thinking I am going to fail.",
    ),
    "work": (
        "Work pressure",
        "My workload has increased recently. I have several deadlines "
        "approaching and I feel under constant pressure to complete "
        "everything. I cannot switch off even at night.",
    ),
    "positive": (
        "Everyday positive",
        "I had a productive day today. I completed my work and spent some "
        "time with my friends. I feel positive about tomorrow.",
    ),
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mental Health Risk Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLES
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
    --bg: #F3F6F5;
    --paper: #FFFFFF;
    --ink: #16302B;
    --muted: #5B6E6A;
    --line: #D5DEDB;
    --accent: #0F766E;
    --accent-dark: #0B5F58;
    --good: #2F855A;
    --warn: #B7791F;
    --bad: #B4443C;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp p, .stApp li, .stApp label { font-size: 15px; }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }

[data-testid="stSidebar"] {
    background: #E9EFED;
    border-right: 1px solid var(--line);
}

/* ---------- header ---------- */
.masthead {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 24px; flex-wrap: wrap;
    padding-bottom: 18px; margin-bottom: 8px;
    border-bottom: 1px solid var(--line);
}
.title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 36px; line-height: 1.12; font-weight: 600;
    color: var(--ink); letter-spacing: -0.01em;
}
.subtitle { color: var(--muted); font-size: 16px; margin-top: 6px; max-width: 62ch; }

.h3 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 23px; font-weight: 600; color: var(--ink);
    margin: 6px 0 2px 0;
}
.lede { color: var(--muted); font-size: 15px; margin-bottom: 12px; max-width: 70ch; }

/* ---------- pills ---------- */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12.5px; font-weight: 500; border: 1px solid var(--line);
    background: var(--paper); color: var(--muted); white-space: nowrap;
}
.pill.good { color: var(--good); background: #E6F3EB; border-color: #BFE0CD; }
.pill.warn { color: var(--warn); background: #FBF1DC; border-color: #EBD3A1; }
.pill.bad  { color: var(--bad);  background: #F9E7E5; border-color: #E9BDB8; }
.pill.info { color: var(--accent); background: #E1F0EE; border-color: #B7DAD5; }

/* ---------- result panel ---------- */
.panel {
    background: var(--paper); border: 1px solid var(--line);
    border-radius: 14px; padding: 22px;
}
.panel.empty {
    border-style: dashed; text-align: center; color: var(--muted);
    padding: 56px 24px;
}
.panel.empty .big {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 22px; color: var(--ink); margin-bottom: 6px;
}

.gauge { width: 100%; max-width: 320px; display: block; margin: 0 auto; }
.gauge-num { fill: var(--ink); font-size: 27px; font-weight: 600; font-family: 'IBM Plex Sans', sans-serif; }
.gauge-cap { fill: var(--muted); font-size: 8.5px; font-family: 'IBM Plex Sans', sans-serif; }
.gauge-fill { animation: gauge-fill 0.9s ease-out; }
@keyframes gauge-fill {
    from { stroke-dasharray: 0px 251.33px; }
    to   { stroke-dasharray: var(--len) 251.33px; }
}
@media (prefers-reduced-motion: reduce) { .gauge-fill { animation: none; } }

.verdict {
    text-align: center; font-family: 'Newsreader', Georgia, serif;
    font-size: 23px; font-weight: 600; margin: 4px 0 16px 0;
}
.kv {
    display: grid; grid-template-columns: auto 1fr; gap: 10px 16px;
    font-size: 14.5px; align-items: center;
    border-top: 1px solid var(--line); padding-top: 14px;
}
.kv .k { color: var(--muted); }
.kv .v { text-align: right; color: var(--ink); font-weight: 500; }

.meta { color: var(--muted); font-size: 13.5px; margin: 6px 0 12px 0; }
.stale { color: var(--warn); font-size: 13.5px; margin-top: 10px; }

/* ---------- evidence ---------- */
.marked {
    background: var(--paper); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px;
    line-height: 1.9; font-size: 16.5px; color: var(--ink);
}
.marked mark { color: inherit; border-radius: 3px; padding: 1px 3px; }
.legend { color: var(--muted); font-size: 13.5px; margin: 8px 0 0 0; }
.legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin: 0 4px 0 12px; }
.legend .sw:first-child { margin-left: 0; }

.tcol-title { font-weight: 600; font-size: 15px; margin-bottom: 4px; }
.tcol-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 8px; }
.trow {
    display: grid; grid-template-columns: minmax(80px, 130px) 1fr 52px;
    align-items: center; gap: 12px; padding: 5px 0; font-size: 14.5px;
}
.trow .tname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trow .tbar { height: 8px; background: #E4ECEA; border-radius: 4px; overflow: hidden; }
.trow .tbar i { display: block; height: 100%; border-radius: 4px; }
.trow .tval { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }

/* ---------- model tab ---------- */
.flow {
    display: grid; grid-template-columns: repeat(4, 1fr);
    background: var(--paper); border: 1px solid var(--line);
    border-radius: 12px; overflow: hidden; margin: 8px 0 18px 0;
}
.flow > div { padding: 16px 18px; border-right: 1px solid var(--line); }
.flow > div:last-child { border-right: none; }
.flow .fname { font-weight: 600; margin-bottom: 4px; }
.flow .fdesc { color: var(--muted); font-size: 14px; }
@media (max-width: 800px) {
    .flow { grid-template-columns: 1fr; }
    .flow > div { border-right: none; border-bottom: 1px solid var(--line); }
    .flow > div:last-child { border-bottom: none; }
}

.spec { display: grid; grid-template-columns: 200px 1fr; gap: 8px 16px; font-size: 14.5px; margin-bottom: 8px; }
.spec .k { color: var(--muted); }

/* ---------- roadmap ---------- */
.road { border-left: 3px solid var(--c); padding: 4px 0 4px 18px; margin: 20px 0; }
.road .rt { font-weight: 600; font-size: 17px; margin-right: 10px; }
.road .rd { color: var(--muted); margin-top: 4px; max-width: 70ch; }

/* ---------- native widgets ---------- */
.stTextArea textarea {
    background: var(--paper); color: var(--ink);
    border: 1px solid var(--line); border-radius: 12px;
    font-size: 16px; line-height: 1.6; padding: 14px 16px;
}
.stTextArea textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }

.stButton > button {
    border-radius: 10px; border: 1px solid var(--line);
    background: var(--paper); color: var(--ink); font-weight: 500;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent); border-color: var(--accent); color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-dark); border-color: var(--accent-dark); color: #FFFFFF;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 500; color: var(--muted); padding: 10px 14px; }
.stTabs [aria-selected="true"] { color: var(--accent); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }

[data-testid="stMetric"] {
    background: var(--paper); border: 1px solid var(--line);
    border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted); }

.foot {
    color: var(--muted); font-size: 13px; text-align: center;
    border-top: 1px solid var(--line); padding-top: 14px; margin-top: 36px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render an HTML snippet. Lines are joined so Markdown never sees
    indented blocks or blank lines inside the markup."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    names, coef = None, None
    try:
        names = np.asarray(vectorizer.get_feature_names_out())
        coef = np.asarray(model.coef_)[0]
    except Exception:
        names, coef = None, None

    return vectorizer, model, names, coef


try:
    vectorizer, model, FEATURE_NAMES, COEF = load_model()
except Exception as error:
    st.error(
        "The model files could not be loaded. Check that both files exist "
        "in the folder below, then reload the page."
    )
    st.code(f"{MODEL_DIR}\n\n{error}")
    st.stop()

EXPLAINABLE = FEATURE_NAMES is not None and COEF is not None


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("text_input", "")
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("history", [])


# ============================================================
# HELPERS
# ============================================================

def count_words(value: str) -> int:
    return len(value.split())


def data_quality(words: int):
    if words < 8:
        return "Too short", "bad", "Very short text. The score is unreliable; add more detail."
    if words < 25:
        return "Limited", "warn", "Short text. Longer posts (50+ words) give steadier scores."
    return "Good", "good", "Enough text for a stable score."


def score_band(score: float):
    if score < 35:
        return "Low", "good", "#2F855A"
    if score < 65:
        return "Borderline", "warn", "#B7791F"
    return "Elevated", "bad", "#B4443C"


def run_analysis(text: str) -> dict:
    X = vectorizer.transform([text])
    probability = float(model.predict_proba(X)[0][1])

    terms = []
    if EXPLAINABLE:
        coo = X.tocoo()
        if coo.nnz:
            contributions = coo.data * COEF[coo.col]
            terms = sorted(
                ((str(FEATURE_NAMES[c]), float(v)) for c, v in zip(coo.col, contributions)),
                key=lambda item: item[1],
            )

    return {
        "text": text,
        "probability": probability,
        "terms": terms,
        "words": count_words(text),
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def gauge_svg(score: float, threshold: float, color: str) -> str:
    length = 251.33  # length of the semicircle (pi * 80)
    filled = length * score / 100
    phi = math.pi * threshold
    x1, y1 = 100 - 66 * math.cos(phi), 100 - 66 * math.sin(phi)
    x2, y2 = 100 - 94 * math.cos(phi), 100 - 94 * math.sin(phi)
    return (
        '<svg viewBox="0 0 200 122" class="gauge" role="img" '
        f'aria-label="Model score {score:.1f} percent">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" stroke="#E4ECEA" '
        'stroke-width="14" fill="none" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" stroke="{color}" '
        'stroke-width="14" fill="none" stroke-linecap="round" class="gauge-fill" '
        f'stroke-dasharray="{filled:.2f} {length}" style="--len:{filled:.2f}px"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        'stroke="#16302B" stroke-width="2"/>'
        f'<text x="100" y="88" text-anchor="middle" class="gauge-num">{score:.1f}%</text>'
        '<text x="100" y="106" text-anchor="middle" class="gauge-cap">Class 1 model score</text>'
        '</svg>'
    )


def highlight_text(text: str, terms) -> str:
    weights = dict(terms)
    if not weights:
        return escape(text).replace("\n", "<br>").replace("$", "&#36;")
    scale = max(abs(v) for v in weights.values()) or 1.0

    pieces = []
    for part in re.split(r"(\W+)", text):
        weight = weights.get(part.lower())
        safe = escape(part).replace("\n", "<br>").replace("$", "&#36;")
        if weight is None or not part.strip():
            pieces.append(safe)
            continue
        alpha = 0.14 + 0.5 * abs(weight) / scale
        rgb = "180,68,60" if weight > 0 else "47,133,90"
        pieces.append(
            f'<mark style="background:rgba({rgb},{alpha:.2f})" '
            f'title="{weight:+.3f}">{safe}</mark>'
        )
    return "".join(pieces)


def term_rows(items, color: str, scale: float) -> str:
    if not items:
        return '<div class="tcol-sub">No terms in this group.</div>'
    rows = []
    for term, value in items:
        width = min(abs(value) / scale * 100, 100) if scale else 0
        rows.append(
            '<div class="trow">'
            f'<span class="tname" title="{escape(term)}">{escape(term)}</span>'
            f'<span class="tbar"><i style="width:{width:.1f}%;background:{color}"></i></span>'
            f'<span class="tval">{value:+.2f}</span>'
            '</div>'
        )
    return "".join(rows)


@st.cache_data(show_spinner=False)
def global_terms(n: int):
    order = np.argsort(COEF)
    negative = [(str(FEATURE_NAMES[i]), float(COEF[i])) for i in order[:n]]
    positive = [(str(FEATURE_NAMES[i]), float(COEF[i])) for i in order[::-1][:n]]
    return positive, negative


# ---- callbacks (run before the next rerun, so they can set widget state)

def use_example(key: str) -> None:
    st.session_state.text_input = EXAMPLES[key][1]
    st.session_state.prediction_result = None
    st.session_state.auto_run = True


def clear_text() -> None:
    st.session_state.text_input = ""
    st.session_state.prediction_result = None


def reset_history() -> None:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Settings")

    threshold = st.slider(
        "Decision threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.01,
        key="threshold",
        help=(
            "Scores at or above this value are labelled Class 1 "
            "(stress-related signal). Lower it to catch more posts; "
            "raise it to reduce false alarms."
        ),
    )

    show_highlight = st.toggle(
        "Highlight terms in the text",
        value=True,
        disabled=not EXPLAINABLE,
    )

    top_n = st.slider(
        "Terms shown per side",
        min_value=3,
        max_value=10,
        value=6,
        disabled=not EXPLAINABLE,
    )

    st.divider()

    history = st.session_state.history
    st.markdown("### This session")
    side1, side2 = st.columns(2)
    side1.metric("Analyses", len(history))
    side2.metric(
        "Mean score",
        f"{np.mean([h['Score (%)'] for h in history]):.0f}%" if history else "-",
    )
    st.button(
        "Reset session history",
        on_click=reset_history,
        use_container_width=True,
        disabled=not history,
    )

    st.caption(
        "Analyzed text is kept in memory for this browser session only. "
        "The history stores word counts and scores, never the text."
    )


# ============================================================
# HEADER
# ============================================================

html(
    f"""
    <div class="masthead">
        <div>
            <div class="title">Early Detection of Mental Health Risk</div>
            <div class="subtitle">A research prototype that screens text for stress-related
            language and shows which words shaped the score.</div>
        </div>
        <div><span class="pill info">Research prototype v{APP_VERSION}</span>
        <span class="pill">Not a diagnostic tool</span></div>
    </div>
    """
)

tab_analyze, tab_model, tab_roadmap, tab_about = st.tabs(
    ["Analyze text", "Model", "Roadmap", "About and responsible use"]
)


# ============================================================
# TAB 1 - ANALYZE
# ============================================================

with tab_analyze:

    left, right = st.columns([1.25, 1], gap="large")

    # ------------------------------------------------ input column
    with left:
        html(
            """
            <div class="h3">Text sample</div>
            <div class="lede">Paste or type a passage, then run the analysis.
            Longer, first-person writing works best.</div>
            """
        )

        st.text_area(
            "Text input",
            key="text_input",
            height=210,
            max_chars=5000,
            placeholder=(
                "Example: I have been feeling overwhelmed with my workload "
                "and I am finding it difficult to concentrate."
            ),
            label_visibility="collapsed",
        )

        current_text = st.session_state.text_input.strip()
        words = count_words(current_text)
        q_label, q_class, q_note = data_quality(words)

        if current_text:
            html(
                f"""
                <div class="meta">{words} words &nbsp;|&nbsp;
                {len(current_text)} of 5000 characters &nbsp;
                <span class="pill {q_class}">Text length: {q_label}</span></div>
                """
            )
        else:
            html('<div class="meta">Nothing entered yet.</div>')

        b1, b2, _spacer = st.columns([2, 1, 2])
        with b1:
            analyze_clicked = st.button(
                "Analyze text", type="primary", use_container_width=True
            )
        with b2:
            st.button("Clear", on_click=clear_text, use_container_width=True)

        st.markdown("**Try an example**")
        ex_cols = st.columns(len(EXAMPLES))
        for col, (key, (label, _)) in zip(ex_cols, EXAMPLES.items()):
            with col:
                st.button(
                    label,
                    key=f"ex_{key}",
                    on_click=use_example,
                    args=(key,),
                    use_container_width=True,
                )

        # ---- run the analysis (button click or one-click example)
        auto_run = st.session_state.pop("auto_run", False)

        if analyze_clicked or auto_run:
            if not current_text:
                st.warning("Enter some text first, then select Analyze text.")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        result = run_analysis(current_text)
                    except Exception as error:
                        st.error("The analysis failed. Details are below.")
                        st.code(str(error))
                    else:
                        st.session_state.prediction_result = result
                        st.session_state.history.append(
                            {
                                "Time": result["time"],
                                "Words": result["words"],
                                "Score (%)": round(result["probability"] * 100, 2),
                            }
                        )
                        st.toast("Analysis complete", icon="✅")

    # ------------------------------------------------ result column
    result = st.session_state.prediction_result

    with right:
        if result is None:
            html(
                """
                <div class="panel empty">
                    <div class="big">No result yet</div>
                    <div>Enter a text sample and select Analyze text, or
                    choose one of the examples.</div>
                </div>
                """
            )
        else:
            probability = result["probability"]
            score = probability * 100
            predicted = probability >= threshold
            band, band_class, band_color = score_band(score)
            r_label, r_class, r_note = data_quality(result["words"])

            verdict = (
                "Stress-related signal detected"
                if predicted
                else "No stress-related signal detected"
            )
            verdict_color = "#B4443C" if predicted else "#2F855A"

            html(
                f"""
                <div class="panel">
                    {gauge_svg(score, threshold, band_color)}
                    <div class="verdict" style="color:{verdict_color}">{verdict}</div>
                    <div class="kv">
                        <span class="k">Classification</span>
                        <span class="v">{"Class 1" if predicted else "Class 0"}
                        (threshold {threshold:.2f})</span>
                        <span class="k">Score band</span>
                        <span class="v"><span class="pill {band_class}">{band}</span></span>
                        <span class="k">Text length</span>
                        <span class="v"><span class="pill {r_class}">{r_label}</span></span>
                        <span class="k">Analyzed at</span>
                        <span class="v">{result["time"]}</span>
                    </div>
                </div>
                """
            )

            if r_class != "good":
                st.caption(r_note)

            if result["text"] != current_text:
                html(
                    '<div class="stale">The text has changed since this result. '
                    "Select Analyze text to update it.</div>"
                )

            st.caption(
                "The tick on the dial marks the decision threshold. The score "
                "is a model-generated Class 1 score for the Dreaddit "
                "stress-classification task. It is not a clinical probability."
            )

    # ------------------------------------------------ evidence section
    if result is not None and EXPLAINABLE and result["terms"]:
        st.divider()

        html(
            """
            <div class="h3">What shaped this score</div>
            <div class="lede">Each term is weighted by how much it appears in the
            text and how strongly the model associates it with stress-related
            posts. This shows what the model responded to. It does not explain
            why a person feels a certain way.</div>
            """
        )

        terms = result["terms"]

        if show_highlight:
            html(f'<div class="marked">{highlight_text(result["text"], terms)}</div>')
            html(
                """
                <div class="legend">
                    <span class="sw" style="background:rgba(180,68,60,0.45)"></span>pushes toward a stress-related signal
                    <span class="sw" style="background:rgba(47,133,90,0.45)"></span>pushes away from it
                    <span class="sw" style="background:#E4ECEA"></span>not scored by the model
                </div>
                """
            )
            st.write("")

        toward = [t for t in reversed(terms) if t[1] > 0][:top_n]
        away = [t for t in terms if t[1] < 0][:top_n]
        scale = max([abs(v) for _, v in toward + away] or [1.0])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            html(
                '<div class="tcol-title">Toward a stress-related signal</div>'
                '<div class="tcol-sub">Higher values pushed the score up.</div>'
                + term_rows(toward, "#B4443C", scale)
            )
        with c2:
            html(
                '<div class="tcol-title">Away from a stress-related signal</div>'
                '<div class="tcol-sub">Lower values pushed the score down.</div>'
                + term_rows(away, "#2F855A", scale)
            )

    elif result is not None and EXPLAINABLE and not result["terms"]:
        st.info(
            "None of the words in this text are in the model's vocabulary, "
            "so there is no term-level evidence to show."
        )

    # ------------------------------------------------ history
    if st.session_state.history:
        st.divider()
        with st.expander(f"Session history ({len(st.session_state.history)})"):
            frame = pd.DataFrame(st.session_state.history)
            frame.index = range(1, len(frame) + 1)
            frame.index.name = "Run"

            if len(frame) > 1:
                st.line_chart(frame["Score (%)"], height=180, color="#0F766E")

            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Download history (CSV)",
                data=frame.to_csv().encode("utf-8"),
                file_name="analysis_history.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 2 - MODEL
# ============================================================

with tab_model:

    html(
        """
        <div class="h3">Text baseline model</div>
        <div class="lede">A transparent baseline: every score can be traced back
        to individual words and their learned weights.</div>
        """
    )

    m_cols = st.columns(len(VALIDATION_METRICS))
    for col, (label, value) in zip(m_cols, VALIDATION_METRICS.items()):
        col.metric(label, value)

    html(
        """
        <div class="flow">
            <div><div class="fname">Text input</div>
            <div class="fdesc">Raw passage, up to 5000 characters.</div></div>
            <div><div class="fname">TF-IDF vectorizer</div>
            <div class="fdesc">Turns words into weighted numeric features.</div></div>
            <div><div class="fname">Logistic regression</div>
            <div class="fdesc">Learns a weight per term for the stress class.</div></div>
            <div><div class="fname">Score and evidence</div>
            <div class="fdesc">Class 1 score plus the terms that drove it.</div></div>
        </div>
        <div class="spec">
            <span class="k">Model</span><span>TF-IDF + Logistic Regression</span>
            <span class="k">Dataset</span><span>Dreaddit (stress classification)</span>
            <span class="k">Validation strategy</span><span>Post-aware split</span>
            <span class="k">Output</span><span>Class 1 score, class label, term-level evidence</span>
        </div>
        """
    )

    if EXPLAINABLE:
        st.divider()
        html(
            """
            <div class="h3">Most influential terms overall</div>
            <div class="lede">The terms the model weights most heavily across the
            whole vocabulary, independent of any single text.</div>
            """
        )

        n_terms = st.slider("Number of terms", 5, 25, 10, key="global_n")
        positive, negative = global_terms(n_terms)
        g_scale = max(abs(v) for _, v in positive + negative) or 1.0

        g1, g2 = st.columns(2, gap="large")
        with g1:
            html(
                '<div class="tcol-title">Strongest stress-related terms</div>'
                '<div class="tcol-sub">Learned coefficient, higher is stronger.</div>'
                + term_rows(positive, "#B4443C", g_scale)
            )
        with g2:
            html(
                '<div class="tcol-title">Strongest non-stress terms</div>'
                '<div class="tcol-sub">Learned coefficient, lower is stronger.</div>'
                + term_rows(negative, "#2F855A", g_scale)
            )

    st.divider()
    st.markdown("**Known limitations**")
    st.markdown(
        "- Trained on Reddit posts, so it may not transfer to other writing styles or languages.\n"
        "- A word-based model cannot read context, sarcasm, or negation reliably.\n"
        "- The score reflects patterns in the training data, not a person's health.\n"
        "- Very short text gives unstable scores."
    )


# ============================================================
# TAB 3 - ROADMAP
# ============================================================

with tab_roadmap:

    html(
        """
        <div class="h3">Multimodal roadmap</div>
        <div class="lede">The current build covers text. Voice and behavior are
        planned extensions of the same framework.</div>
        """
    )

    ROADMAP = [
        ("Text", "Working", "good", "#2F855A",
         "TF-IDF and logistic regression baseline on Dreaddit, with term-level evidence for each score."),
        ("Voice", "In progress", "warn", "#B7791F",
         "Extending the framework to speech features using the controlled-access MODMA dataset."),
        ("Behavior", "Planned", "info", "#0F766E",
         "Longitudinal behavioral and time-series signals under the same data schema."),
        ("Multimodal fusion", "Planned", "info", "#0F766E",
         "Combine text, voice, and behavior with uncertainty estimates and human-review recommendations."),
    ]

    for name, status, css_class, color, description in ROADMAP:
        html(
            f"""
            <div class="road" style="--c:{color}">
                <div><span class="rt">{name}</span>
                <span class="pill {css_class}">{status}</span></div>
                <div class="rd">{description}</div>
            </div>
            """
        )


# ============================================================
# TAB 4 - ABOUT
# ============================================================

with tab_about:

    html(
        """
        <div class="h3">About this project</div>
        <div class="lede">A privacy-aware data science framework for identifying
        early mental-health-related risk signals. It is a screening and
        referral-support research prototype, not a clinical system.</div>
        """
    )

    st.markdown("**Responsible use**")
    st.markdown(
        "- Outputs are research classification signals learned from the Dreaddit dataset.\n"
        "- A score is not a diagnosis and should never replace professional assessment.\n"
        "- Do not use this tool to make decisions about a real person's care, employment, or education.\n"
        "- Text you enter is processed in memory during your session and is not saved by this app."
    )

    st.info(
        "If you or someone you know is struggling, please contact a qualified "
        "mental health professional, a trusted person, or your local emergency "
        "services."
    )

    st.markdown("**Project team**")
    st.markdown(
        "Kunwar Aditya Singh and Mohammad Shahzan  \n"
        "B.Tech (AI and DS), Khwaja Moinuddin Chishti Language University, "
        "Lucknow, Uttar Pradesh  \n"
        "Academic year 2026-27"
    )


# ============================================================
# DISCLAIMER AND FOOTER
# ============================================================

st.warning(
    "Research-only output. This system produces a classification signal learned "
    "from the Dreaddit stress dataset. It is not a medical diagnosis or a "
    "clinical probability and should not replace professional assessment."
)

html(
    f"""
    <div class="foot">Data Science Framework for Early Detection of Mental Health Risk
    &nbsp;|&nbsp; Research prototype v{APP_VERSION}</div>
    """
)