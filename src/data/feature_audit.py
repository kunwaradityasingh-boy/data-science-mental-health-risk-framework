from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.dreaddit_features import (
    get_engineered_feature_columns,
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a Dreaddit CSV dataset."""
    return pd.read_csv(path)


def audit_feature_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Audit engineered features for:
    - feature count
    - missing values
    - constant features
    - infinite values
    - train/test distribution statistics
    """

    train_features = get_engineered_feature_columns(train_df)
    test_features = get_engineered_feature_columns(test_df)

    print("=" * 70)
    print("DREAddit FEATURE LEAKAGE & DISTRIBUTION AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Feature count
    # ---------------------------------------------------------

    print("\n[1] FEATURE COUNT")

    print(f"Train engineered features : {len(train_features)}")
    print(f"Test engineered features  : {len(test_features)}")

    if train_features != test_features:
        raise ValueError(
            "Train and test engineered feature columns do not match."
        )

    print("[OK] Train/test feature columns match.")

    # ---------------------------------------------------------
    # 2. Missing values
    # ---------------------------------------------------------

    print("\n[2] MISSING VALUES")

    train_missing = train_df[train_features].isna().sum().sum()
    test_missing = test_df[test_features].isna().sum().sum()

    print(f"Train missing values : {train_missing}")
    print(f"Test missing values  : {test_missing}")

    if train_missing == 0 and test_missing == 0:
        print("[OK] No missing engineered feature values.")

    # ---------------------------------------------------------
    # 3. Infinite values
    # ---------------------------------------------------------

    print("\n[3] INFINITE VALUES")

    train_inf = (~train_df[train_features].apply(
        lambda column: column.map(pd.api.types.is_number)
    )).sum().sum()

    # Direct numeric infinity check
    train_numeric = train_df[train_features]
    test_numeric = test_df[test_features]

    train_inf_count = train_numeric.map(
        lambda value: value == float("inf")
        or value == float("-inf")
    ).sum().sum()

    test_inf_count = test_numeric.map(
        lambda value: value == float("inf")
        or value == float("-inf")
    ).sum().sum()

    print(f"Train infinite values : {train_inf_count}")
    print(f"Test infinite values  : {test_inf_count}")

    if train_inf_count == 0 and test_inf_count == 0:
        print("[OK] No infinite engineered feature values.")

    # ---------------------------------------------------------
    # 4. Constant features
    # ---------------------------------------------------------

    print("\n[4] CONSTANT FEATURES")

    constant_features = []

    for column in train_features:
        if train_df[column].nunique(dropna=False) <= 1:
            constant_features.append(column)

    print(
        f"Constant features in training data: "
        f"{len(constant_features)}"
    )

    if constant_features:
        print("Constant feature names:")
        for column in constant_features:
            print(f"  - {column}")
    else:
        print("[OK] No constant engineered features.")

    # ---------------------------------------------------------
    # 5. Target leakage candidates
    # ---------------------------------------------------------

    print("\n[5] TARGET LEAKAGE CANDIDATES")

    suspicious_names = []

    leakage_keywords = [
        "label",
        "target",
        "class",
        "diagnosis",
        "outcome",
        "risk",
    ]

    for column in train_features:
        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in leakage_keywords
        ):
            suspicious_names.append(column)

    if suspicious_names:
        print("Potentially suspicious feature names:")
        for column in suspicious_names:
            print(f"  - {column}")
    else:
        print(
            "[OK] No engineered feature names directly indicate "
            "target/diagnosis/risk."
        )

    # ---------------------------------------------------------
    # 6. Train/Test distribution summary
    # ---------------------------------------------------------

    print("\n[6] TRAIN/TEST DISTRIBUTION SUMMARY")

    summary_rows = []

    for column in train_features:

        train_mean = train_df[column].mean()
        test_mean = test_df[column].mean()

        train_std = train_df[column].std()
        test_std = test_df[column].std()

        summary_rows.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary["mean_difference"] = (
        summary["test_mean"]
        - summary["train_mean"]
    ).abs()

    summary["std_difference"] = (
        summary["test_std"]
        - summary["train_std"]
    ).abs()

    print(
        summary[
            [
                "feature",
                "train_mean",
                "test_mean",
                "mean_difference",
            ]
        ]
        .sort_values(
            "mean_difference",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(
        "\n[INFO] The distribution table is descriptive only. "
        "Large differences require investigation but do not "
        "automatically prove leakage."
    )

    print("\n" + "=" * 70)
    print("FEATURE AUDIT COMPLETE")
    print("=" * 70)


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    data_dir = (
        base_dir
        / "data"
        / "raw"
        / "dreaddit"
        / "extracted"
    )

    train_path = data_dir / "dreaddit-train.csv"
    test_path = data_dir / "dreaddit-test.csv"

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    audit_feature_columns(
        train_df,
        test_df,
    )


if __name__ == "__main__":
    main()