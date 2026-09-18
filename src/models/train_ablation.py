from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import (
    get_engineered_feature_columns,
)
from src.features.feature_groups import (
    group_features,
)
from src.features.tfidf_features import (
    fit_tfidf,
)


def evaluate_model(
    X_train,
    X_validation,
    y_train,
    y_validation,
):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    return {
        "accuracy": accuracy_score(
            y_validation,
            predictions,
        ),
        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
    }


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

    report_dir = (
        base_dir
        / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )

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

    train_df = df.iloc[
        train_indices
    ].copy()

    validation_df = df.iloc[
        validation_indices
    ].copy()

    y_train = train_df["label"]
    y_validation = validation_df["label"]

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    vectorizer, X_train_text = fit_tfidf(
        train_df["text"]
    )

    X_validation_text = vectorizer.transform(
        validation_df["text"]
    )

    # ---------------------------------------------------------
    # Engineered feature groups
    # ---------------------------------------------------------

    train_features = get_engineered_feature_columns(
        train_df
    )

    validation_features = get_engineered_feature_columns(
        validation_df
    )

    if train_features != validation_features:
        raise ValueError(
            "Train and validation engineered feature "
            "columns do not match."
        )

    grouped = group_features(
        train_df
    )

    # ---------------------------------------------------------
    # Experiments
    # ---------------------------------------------------------

    experiments = {
        "text_only": [],
        "text_plus_liwc": grouped.get(
            "liwc",
            []
        ),
        "text_plus_dal": grouped.get(
            "dal",
            []
        ),
        "text_plus_social": grouped.get(
            "social",
            []
        ),
        "text_plus_syntax_readability": grouped.get(
            "syntax_readability",
            []
        ),
        "text_plus_sentiment": grouped.get(
            "sentiment",
            []
        ),
        "text_plus_all_engineered": train_features,
    }

    results = []

    print("=" * 70)
    print("DREAddit ABLATION STUDY")
    print("=" * 70)

    print(
        f"Training records   : {len(train_df)}"
    )

    print(
        f"Validation records : {len(validation_df)}"
    )

    print(
        f"TF-IDF features    : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print()

    # ---------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------

    for experiment_name, selected_features in experiments.items():

        if not selected_features:

            X_train = X_train_text
            X_validation = X_validation_text

            engineered_count = 0

        else:

            X_train_engineered = train_df[
                selected_features
            ]

            X_validation_engineered = validation_df[
                selected_features
            ]

            scaler = StandardScaler()

            X_train_engineered_scaled = (
                scaler.fit_transform(
                    X_train_engineered
                )
            )

            X_validation_engineered_scaled = (
                scaler.transform(
                    X_validation_engineered
                )
            )

            X_train_engineered_sparse = csr_matrix(
                X_train_engineered_scaled
            )

            X_validation_engineered_sparse = csr_matrix(
                X_validation_engineered_scaled
            )

            X_train = hstack(
                [
                    X_train_text,
                    X_train_engineered_sparse,
                ],
                format="csr",
            )

            X_validation = hstack(
                [
                    X_validation_text,
                    X_validation_engineered_sparse,
                ],
                format="csr",
            )

            engineered_count = len(
                selected_features
            )

        metrics = evaluate_model(
            X_train,
            X_validation,
            y_train,
            y_validation,
        )

        result = {
            "experiment": experiment_name,
            "engineered_features": engineered_count,
            "total_features": X_train.shape[1],
            **metrics,
        }

        results.append(
            result
        )

        print(
            f"{experiment_name:35s}"
            f" | engineered={engineered_count:3d}"
            f" | accuracy={metrics['accuracy']:.4f}"
            f" | f1={metrics['f1']:.4f}"
            f" | roc_auc={metrics['roc_auc']:.4f}"
            f" | pr_auc={metrics['pr_auc']:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        report_dir
        / "ablation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"[OK] Ablation results saved to:"
    )
    print(output_path)

    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()