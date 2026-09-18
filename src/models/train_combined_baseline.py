from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import (
    extract_engineered_features,
)
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

    model_dir = (
        base_dir
        / "models"
        / "combined_baseline"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(dataset_path)

    # ---------------------------------------------------------
    # Post-aware train/validation split
    # ---------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, validation_indices = next(
        splitter.split(
            df,
            groups=df["post_id"],
        )
    )

    train_df = df.iloc[train_indices].copy()
    validation_df = df.iloc[validation_indices].copy()

    y_train = train_df["label"]
    y_validation = validation_df["label"]

    # ---------------------------------------------------------
    # TF-IDF text features
    # ---------------------------------------------------------

    vectorizer, X_train_text = fit_tfidf(
        train_df["text"]
    )

    X_validation_text = vectorizer.transform(
        validation_df["text"]
    )

    # ---------------------------------------------------------
    # Engineered features
    # ---------------------------------------------------------

    X_train_engineered = extract_engineered_features(
        train_df
    )

    X_validation_engineered = extract_engineered_features(
        validation_df
    )

    # ---------------------------------------------------------
    # Standardize engineered features
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_engineered_scaled = scaler.fit_transform(
        X_train_engineered
    )

    X_validation_engineered_scaled = scaler.transform(
        X_validation_engineered
    )

    # Convert to sparse matrices
    X_train_engineered_sparse = csr_matrix(
        X_train_engineered_scaled
    )

    X_validation_engineered_sparse = csr_matrix(
        X_validation_engineered_scaled
    )

    # ---------------------------------------------------------
    # Combine text + engineered features
    # ---------------------------------------------------------

    X_train_combined = hstack(
        [
            X_train_text,
            X_train_engineered_sparse,
        ],
        format="csr",
    )

    X_validation_combined = hstack(
        [
            X_validation_text,
            X_validation_engineered_sparse,
        ],
        format="csr",
    )

    # ---------------------------------------------------------
    # Train Logistic Regression
    # ---------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_combined,
        y_train,
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_validation_combined
    )

    probabilities = model.predict_proba(
        X_validation_combined
    )[:, 1]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("=" * 70)
    print("COMBINED TEXT + ENGINEERED BASELINE")
    print("=" * 70)

    print(
        f"Training records        : {len(train_df)}"
    )

    print(
        f"Validation records      : {len(validation_df)}"
    )

    print(
        f"TF-IDF features         : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Engineered features     : "
        f"{X_train_engineered.shape[1]}"
    )

    print(
        f"Combined features       : "
        f"{X_train_combined.shape[1]}"
    )

    print()
    print(f"Accuracy                : {accuracy:.4f}")
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1                      : {f1:.4f}")
    print(f"ROC-AUC                 : {roc_auc:.4f}")
    print(f"PR-AUC                  : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_validation,
            predictions,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Save artifacts
    # ---------------------------------------------------------

    joblib.dump(
        vectorizer,
        model_dir / "tfidf_vectorizer.joblib",
    )

    joblib.dump(
        scaler,
        model_dir / "engineered_scaler.joblib",
    )

    joblib.dump(
        model,
        model_dir / "logistic_regression.joblib",
    )

    print("\nArtifacts saved:")
    print(
        model_dir
        / "tfidf_vectorizer.joblib"
    )

    print(
        model_dir
        / "engineered_scaler.joblib"
    )

    print(
        model_dir
        / "logistic_regression.joblib"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()