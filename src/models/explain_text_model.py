from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.tfidf_features import fit_tfidf


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]

    dataset_path = (
        base_dir
        / "data"
        / "raw"
        / "dreaddit"
        / "extracted"
        / "dreaddit-train.csv"
    )

    report_dir = base_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

    df = pd.read_csv(dataset_path)

    X_text = df["text"]
    y = df["label"]

    print("=" * 70)
    print("TEXT MODEL EXPLAINABILITY")
    print("=" * 70)

    print(f"Training records : {len(df)}")

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    vectorizer, X = fit_tfidf(X_text)

    print(
        f"TF-IDF vocabulary : {len(vectorizer.vocabulary_)}"
    )

    # ---------------------------------------------------------
    # Train baseline model
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X, y)

    # ---------------------------------------------------------
    # Extract feature names and coefficients
    # ---------------------------------------------------------

    feature_names = vectorizer.get_feature_names_out()

    coefficients = model.coef_[0]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    # Positive coefficient -> Class 1
    # Negative coefficient -> Class 0

    positive_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(30)
        .copy()
    )

    negative_features = (
        importance_df
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(30)
        .copy()
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    positive_path = (
        report_dir
        / "text_model_positive_features.csv"
    )

    negative_path = (
        report_dir
        / "text_model_negative_features.csv"
    )

    all_path = (
        report_dir
        / "text_model_coefficients.csv"
    )

    positive_features.to_csv(
        positive_path,
        index=False,
    )

    negative_features.to_csv(
        negative_path,
        index=False,
    )

    importance_df.to_csv(
        all_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP FEATURES CONTRIBUTING TOWARD CLASS 1")
    print("=" * 70)

    print(
        positive_features.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 70)
    print("TOP FEATURES CONTRIBUTING TOWARD CLASS 0")
    print("=" * 70)

    print(
        negative_features.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(positive_path)
    print(negative_path)
    print(all_path)

    print()
    print("=" * 70)
    print("EXPLAINABILITY STEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()