from pathlib import Path

import pandas as pd
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import extract_engineered_features


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "dreaddit"
    / "extracted"
    / "dreaddit-train.csv"
)


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
    print("ENGINEERED FEATURES BASELINE")
    print("109 FEATURES + LOGISTIC REGRESSION")
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
    # 3. Extract engineered features
    # ---------------------------------------------------------

    X_train = extract_engineered_features(train_df)
    X_validation = extract_engineered_features(validation_df)

    y_train = train_df["label"]
    y_validation = validation_df["label"]

    print(f"Engineered feature count: {X_train.shape[1]}")
    print(f"Training feature shape: {X_train.shape}")
    print(f"Validation feature shape: {X_validation.shape}")

    # ---------------------------------------------------------
    # 4. Build model pipeline
    # ---------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 6. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ---------------------------------------------------------
    # 7. Metrics
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
    # 8. Results
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
    # 9. Confusion matrix
    # ---------------------------------------------------------

    print("\nConfusion Matrix")
    print("-" * 60)

    print(
        confusion_matrix(
            y_validation,
            predictions,
        )
    )

    # ---------------------------------------------------------
    # 10. Classification report
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
    # 11. Leakage check
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

    print(
        "[OK] Engineered feature baseline completed."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()