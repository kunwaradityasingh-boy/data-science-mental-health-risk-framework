from __future__ import annotations

from pathlib import Path

import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
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


SEEDS = [42, 7, 21, 100, 123]


def evaluate(
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

    report_dir = base_dir / "reports"

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        dataset_path
    )

    engineered_features = (
        get_engineered_feature_columns(df)
    )

    grouped = group_features(df)

    social_features = grouped.get(
        "social",
        []
    )

    experiments = {
        "text_only": [],
        "text_plus_social": social_features,
        "text_plus_all_engineered": engineered_features,
    }

    results = []

    print("=" * 75)
    print("DREAddit REPEATED GROUP-AWARE VALIDATION")
    print("=" * 75)

    print(
        f"Dataset records       : {len(df)}"
    )

    print(
        f"Engineered features   : {len(engineered_features)}"
    )

    print(
        f"Social features       : {len(social_features)}"
    )

    print(
        f"Validation seeds      : {SEEDS}"
    )

    print()

    # ---------------------------------------------------------
    # Repeat experiments over multiple group-aware splits
    # ---------------------------------------------------------

    for seed in SEEDS:

        print(
            f"Running validation seed: {seed}"
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
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

        # -----------------------------------------------------
        # TF-IDF fit ONLY on this split's training data
        # -----------------------------------------------------

        vectorizer, X_train_text = fit_tfidf(
            train_df["text"]
        )

        X_validation_text = vectorizer.transform(
            validation_df["text"]
        )

        # -----------------------------------------------------
        # Run each experiment
        # -----------------------------------------------------

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

                X_train_scaled = scaler.fit_transform(
                    X_train_engineered
                )

                X_validation_scaled = scaler.transform(
                    X_validation_engineered
                )

                X_train_engineered_sparse = csr_matrix(
                    X_train_scaled
                )

                X_validation_engineered_sparse = csr_matrix(
                    X_validation_scaled
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

            metrics = evaluate(
                X_train,
                X_validation,
                y_train,
                y_validation,
            )

            results.append(
                {
                    "seed": seed,
                    "experiment": experiment_name,
                    "engineered_features": engineered_count,
                    **metrics,
                }
            )

            print(
                f"  {experiment_name:30s}"
                f" | F1={metrics['f1']:.4f}"
                f" | ROC-AUC={metrics['roc_auc']:.4f}"
                f" | PR-AUC={metrics['pr_auc']:.4f}"
            )

    # ---------------------------------------------------------
    # Save individual results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    detailed_path = (
        report_dir
        / "repeated_validation_results.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    summary = (
        results_df
        .groupby("experiment")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
        )
        .reset_index()
    )

    summary_path = (
        report_dir
        / "repeated_validation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("REPEATED VALIDATION SUMMARY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(
        f"[OK] Detailed results saved to:"
    )
    print(detailed_path)

    print(
        f"[OK] Summary results saved to:"
    )
    print(summary_path)

    print("=" * 75)
    print("REPEATED VALIDATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()