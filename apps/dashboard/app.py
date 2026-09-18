import sys
from pathlib import Path

import streamlit as st


# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.predict_text import predict_text


st.set_page_config(
    page_title="Mental Health Risk Research",
    page_icon="🧠",
    layout="centered",
)


st.title("🧠 Mental Health Risk Research Prototype")

st.caption(
    "Privacy-aware text-based research screening prototype"
)


st.warning(
    "Research prototype only. "
    "This system does not diagnose depression, anxiety, "
    "or any other mental-health disorder."
)


st.markdown("### Text Analysis")

text = st.text_area(
    "Enter a text sample for research classification:",
    height=180,
    placeholder=(
        "Example: I have been feeling overwhelmed "
        "and struggling with my daily responsibilities."
    ),
)


analyze = st.button(
    "🔍 Analyze Text",
    type="primary",
)


if analyze:

    if not text.strip():
        st.error("Please enter some text before analysis.")

    else:

        with st.spinner("Analyzing text..."):

            try:
                result = predict_text(text)

                prediction = result["prediction"]
                probability = result["probability_class_1"]
                risk_signal = result["risk_signal"]

                st.markdown("---")

                st.subheader(
                    "Research Classification Result"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Class-1 Score",
                        f"{probability:.2%}",
                    )

                with col2:

                    if prediction == 1:
                        st.metric(
                            "Classification",
                            "Class 1",
                        )
                    else:
                        st.metric(
                            "Classification",
                            "Class 0",
                        )

                st.markdown("### Signal")

                if prediction == 1:
                    st.warning(
                        f"⚠️ {risk_signal}"
                    )
                else:
                    st.info(
                        f"ℹ️ {risk_signal}"
                    )

                st.markdown("### Model Information")

                st.write(
                    "Model: TF-IDF + Logistic Regression"
                )

                st.write(
                    "Dataset: Dreaddit"
                )

                st.write(
                    "Validation strategy: Post-aware split"
                )

                st.write(
                    "Validation Accuracy: 75.26%"
                )

                st.write(
                    "Validation F1 Score: 77.43%"
                )

                st.markdown("### Important Limitation")

                st.info(
                    "The output represents a research classification "
                    "signal learned from the Dreaddit stress dataset. "
                    "It is not a medical diagnosis, clinical probability, "
                    "or substitute for professional assessment."
                )

            except Exception as exc:

                st.error(
                    f"Prediction failed: {exc}"
                )