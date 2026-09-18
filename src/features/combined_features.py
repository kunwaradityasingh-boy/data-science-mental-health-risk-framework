from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse import hstack, csr_matrix

from src.features.tfidf_features import fit_tfidf
from src.features.dreaddit_features import extract_engineered_features


def build_combined_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
):
    """
    Build a combined representation using:

    1. TF-IDF text features
    2. Standardized engineered numeric features

    TF-IDF remains sparse.
    Engineered features are converted to sparse format
    before horizontal concatenation.
    """

    # ---------------------------------------------------------
    # 1. Text features
    # ---------------------------------------------------------

    vectorizer, X_train_text = fit_tfidf(
        train_df["text"]
    )

    X_validation_text = vectorizer.transform(
        validation_df["text"]
    )

    # ---------------------------------------------------------
    # 2. Engineered features
    # ---------------------------------------------------------

    X_train_engineered = extract_engineered_features(
        train_df
    )

    X_validation_engineered = extract_engineered_features(
        validation_df
    )

    # ---------------------------------------------------------
    # 3. Convert engineered features to sparse matrices
    # ---------------------------------------------------------

    X_train_engineered_sparse = csr_matrix(
        X_train_engineered.to_numpy()
    )

    X_validation_engineered_sparse = csr_matrix(
        X_validation_engineered.to_numpy()
    )

    # ---------------------------------------------------------
    # 4. Combine text + engineered features
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

    return (
        vectorizer,
        X_train_combined,
        X_validation_combined,
    )


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

    df = pd.read_csv(dataset_path)

    # ---------------------------------------------------------
    # Same post-aware validation split used previously
    # ---------------------------------------------------------

    from sklearn.model_selection import GroupShuffleSplit

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

    # ---------------------------------------------------------
    # Build combined matrix
    # ---------------------------------------------------------

    (
        vectorizer,
        X_train_combined,
        X_validation_combined,
    ) = build_combined_features(
        train_df,
        validation_df,
    )

    print("=" * 70)
    print("COMBINED TEXT + ENGINEERED FEATURE MATRIX")
    print("=" * 70)

    print(
        f"Training records       : {len(train_df)}"
    )

    print(
        f"Validation records     : {len(validation_df)}"
    )

    print(
        f"TF-IDF vocabulary      : {len(vectorizer.vocabulary_)}"
    )

    print(
        f"Combined train shape   : {X_train_combined.shape}"
    )

    print(
        f"Combined validation shape: "
        f"{X_validation_combined.shape}"
    )

    print(
        f"Engineered features    : "
        f"{X_train_combined.shape[1] - len(vectorizer.vocabulary_)}"
    )

    print("=" * 70)

    print(
        "\n[OK] Combined feature matrix created successfully."
    )


if __name__ == "__main__":
    main()