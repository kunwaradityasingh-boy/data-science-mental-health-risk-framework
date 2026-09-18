from pathlib import Path

from joblib import load

from src.features.tfidf_features import clean_text


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "text_baseline"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def load_artifacts():
    """Load the trained TF-IDF vectorizer and Logistic Regression model."""

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {VECTORIZER_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Logistic Regression model not found: {MODEL_PATH}"
        )

    vectorizer = load(VECTORIZER_PATH)
    model = load(MODEL_PATH)

    return vectorizer, model


def predict_text(text: str) -> dict:
    """
    Predict the Dreaddit stress-related research label
    for a single text input.

    This is a research classification signal and NOT
    a clinical diagnosis.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        raise ValueError("text cannot be empty")

    vectorizer, model = load_artifacts()

    cleaned_text = clean_text(text)

    features = vectorizer.transform([cleaned_text])

    prediction = int(model.predict(features)[0])

    probability = float(
        model.predict_proba(features)[0][1]
    )

    if prediction == 1:
        risk_signal = "Stress-related signal detected"
    else:
        risk_signal = "No stress-related signal detected"

    return {
        "prediction": prediction,
        "probability_class_1": probability,
        "risk_signal": risk_signal,
    }


def main():
    print("=" * 60)
    print("TEXT PREDICTION PIPELINE")
    print("=" * 60)

    sample_text = (
        "I have been feeling overwhelmed with everything "
        "and I am struggling to manage my daily responsibilities."
    )

    result = predict_text(sample_text)

    print("\nInput:")
    print(sample_text)

    print("\nPrediction:")
    print(result["prediction"])

    print("\nClass 1 probability:")
    print(f"{result['probability_class_1']:.4f}")

    print("\nRisk signal:")
    print(result["risk_signal"])

    print("\nDisclaimer:")
    print(
        "Research prototype only. "
        "This output is not a clinical diagnosis."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()