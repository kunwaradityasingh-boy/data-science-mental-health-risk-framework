from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack

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
from sklearn.preprocessing import StandardScaler

from src.features.dreaddit_features import get_engineered_feature_columns
from src.features.feature_groups import group_features
from src.features.tfidf_features import fit_tfidf


def train_and_predict(
    X_train,
    X_test,
    y_train,
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

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return model, predictions, probabilities


def calculate_metrics(
    y_test,
    predictions,
    probabilities,
):
    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    data_dir = (
        base_dir
        / "data"
        / "raw"
        / "dreaddit"
        / "extracted"
    )

    train_path = (
        data_dir
        / "dreaddit-train.csv"
    )

    test_path = (
        data_dir
        / "dreaddit-test.csv"
    )

    report_dir = base_dir / "reports"

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load supplied train/test data
    # ---------------------------------------------------------

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("=" * 75)
    print("DREAddit FINAL HOLDOUT EVALUATION + PREDICTION ARTIFACT")
    print("=" * 75)

    print(
        f"Training records : {len(train_df)}"
    )

    print(
        f"Test records     : {len(test_df)}"
    )

    # ---------------------------------------------------------
    # Engineered feature verification
    # ---------------------------------------------------------

    engineered_features = (
        get_engineered_feature_columns(
            train_df
        )
    )

    test_engineered_features = (
        get_engineered_feature_columns(
            test_df
        )
    )

    if engineered_features != test_engineered_features:
        raise ValueError(
            "Train/test engineered feature columns do not match."
        )

    grouped = group_features(train_df)

    social_features = grouped.get(
        "social",
        []
    )

    experiments = {
        "text_only": [],
        "text_plus_social": social_features,
        "text_plus_all_engineered": engineered_features,
    }

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    vectorizer, X_train_text = fit_tfidf(
        train_df["text"]
    )

    X_test_text = vectorizer.transform(
        test_df["text"]
    )

    print(
        f"TF-IDF vocabulary : "
        f"{len(vectorizer.vocabulary_)}"
    )

    all_results = []
    all_predictions = []

    # ---------------------------------------------------------
    # Run predefined experiments
    # ---------------------------------------------------------

    for experiment_name, selected_features in experiments.items():

        print()
        print("-" * 75)
        print(
            f"EXPERIMENT: {experiment_name}"
        )
        print("-" * 75)

        if not selected_features:

            X_train = X_train_text
            X_test = X_test_text

        else:

            scaler = StandardScaler()

            X_train_engineered = (
                scaler.fit_transform(
                    train_df[
                        selected_features
                    ]
                )
            )

            X_test_engineered = (
                scaler.transform(
                    test_df[
                        selected_features
                    ]
                )
            )

            X_train = hstack(
                [
                    X_train_text,
                    csr_matrix(
                        X_train_engineered
                    ),
                ],
                format="csr",
            )

            X_test = hstack(
                [
                    X_test_text,
                    csr_matrix(
                        X_test_engineered
                    ),
                ],
                format="csr",
            )

        model, predictions, probabilities = (
            train_and_predict(
                X_train,
                X_test,
                y_train,
            )
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        print(
            f"Features : {X_train.shape[1]}"
        )

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1        : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : {metrics['pr_auc']:.4f}"
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                y_test,
                predictions,
            )
        )

        all_results.append(
            {
                "experiment": experiment_name,
                "feature_count": X_train.shape[1],
                **metrics,
            }
        )

        # -----------------------------------------------------
        # Save row-level predictions
        # -----------------------------------------------------

        experiment_predictions = pd.DataFrame(
            {
                "experiment": experiment_name,
                "test_row_index": test_df.index,
                "id": test_df["id"].values,
                "post_id": test_df["post_id"].values,
                "subreddit": test_df["subreddit"].values,
                "actual_label": y_test.values,
                "predicted_label": predictions,
                "class_1_score": probabilities,
                "text_length": test_df[
                    "text"
                ].astype(str).str.len().values,
            }
        )

        experiment_predictions[
            "error_type"
        ] = "correct"

        experiment_predictions.loc[
            (
                experiment_predictions[
                    "actual_label"
                ]
                == 0
            )
            & (
                experiment_predictions[
                    "predicted_label"
                ]
                == 1
            ),
            "error_type",
        ] = "false_positive"

        experiment_predictions.loc[
            (
                experiment_predictions[
                    "actual_label"
                ]
                == 1
            )
            & (
                experiment_predictions[
                    "predicted_label"
                ]
                == 0
            ),
            "error_type",
        ] = "false_negative"

        experiment_predictions.loc[
            (
                experiment_predictions[
                    "actual_label"
                ]
                == 0
            )
            & (
                experiment_predictions[
                    "predicted_label"
                ]
                == 0
            ),
            "error_type",
        ] = "true_negative"

        experiment_predictions.loc[
            (
                experiment_predictions[
                    "actual_label"
                ]
                == 1
            )
            & (
                experiment_predictions[
                    "predicted_label"
                ]
                == 1
            ),
            "error_type",
        ] = "true_positive"

        all_predictions.append(
            experiment_predictions
        )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_path = (
        report_dir
        / "final_holdout_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save all row-level predictions
    # ---------------------------------------------------------

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        report_dir
        / "final_holdout_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL HOLDOUT SUMMARY")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 75)
    print("PREDICTION ARTIFACT")
    print("=" * 75)

    print(
        f"Prediction rows : {len(predictions_df)}"
    )

    print(
        f"Expected rows   : "
        f"{len(test_df) * len(experiments)}"
    )

    print(
        f"Saved to        : {predictions_path}"
    )

    print()
    print("=" * 75)
    print("FINAL HOLDOUT ARTIFACT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()