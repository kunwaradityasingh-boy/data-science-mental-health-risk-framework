from pathlib import Path

import pandas as pd
from joblib import dump
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

from src.features.tfidf_features import build_tfidf_vectorizer


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "dreaddit"
    / "extracted"
    / "dreaddit-train.csv"
)

MODEL_DIR = BASE_DIR / "models" / "text_baseline"


def create_post_aware_split(
    df: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = 42,
):
    """Split data by post_id to prevent post-level leakage."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )

    train_indices, validation_indices = next(
        splitter.split(
            df,
            y=df["label"],
            groups=df["post_id"],
        )
    )

    train_df = df.iloc[train_indices].copy()
    validation_df = df.iloc[validation_indices].copy()

    return train_df, validation_df


def main():
    print("=" * 60)
    print("TEXT BASELINE MODEL EVALUATION")
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    print(f"\nTotal records: {len(df)}")

    # ---------------------------------------------------------
    # 2. Post-aware split
    # ---------------------------------------------------------

    train_df, validation_df = create_post_aware_split(df)

    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")

    # ---------------------------------------------------------
    # 3. Prepare text and labels
    # ---------------------------------------------------------

    X_train_text = train_df["text"].fillna("")
    y_train = train_df["label"]

    X_validation_text = validation_df["text"].fillna("")
    y_validation = validation_df["label"]

    # ---------------------------------------------------------
    # 4. TF-IDF
    # ---------------------------------------------------------

    print("\nFitting TF-IDF...")

    vectorizer = build_tfidf_vectorizer()

    # Fit ONLY on training data
    X_train = vectorizer.fit_transform(X_train_text)

    # Validation is only transformed
    X_validation = vectorizer.transform(X_validation_text)

    print(f"Training TF-IDF shape: {X_train.shape}")
    print(f"Validation TF-IDF shape: {X_validation.shape}")

    # ---------------------------------------------------------
    # 5. Train Logistic Regression
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 6. Save model artifacts
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    model_path = MODEL_DIR / "logistic_regression.joblib"

    dump(
        vectorizer,
        vectorizer_path,
    )

    dump(
        model,
        model_path,
    )

    print("\nModel Artifacts")
    print("-" * 60)
    print(f"TF-IDF vectorizer : {vectorizer_path}")
    print(f"Logistic model    : {model_path}")

    # ---------------------------------------------------------
    # 7. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(X_validation)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ---------------------------------------------------------
    # 8. Metrics
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
    # 9. Print metrics
    # ---------------------------------------------------------

    print("\nValidation Results")
    print("-" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    # ---------------------------------------------------------
    # 10. Confusion Matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_validation,
        predictions,
    )

    print("\nConfusion Matrix")
    print("-" * 60)
    print(cm)

    # ---------------------------------------------------------
    # 11. Classification Report
    # ---------------------------------------------------------

    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_validation,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 12. Leakage validation
    # ---------------------------------------------------------

    train_posts = set(
        train_df["post_id"]
    )

    validation_posts = set(
        validation_df["post_id"]
    )

    overlapping_posts = (
        train_posts & validation_posts
    )

    if overlapping_posts:
        raise RuntimeError(
            "Post leakage detected between "
            "training and validation."
        )

    print(
        "[OK] No post_id overlap between "
        "training and validation."
    )

    print("[OK] TF-IDF vectorizer saved.")
    print("[OK] Logistic Regression model saved.")
    print("[OK] Complete baseline evaluation finished.")

    print("=" * 60)


if __name__ == "__main__":
    main()