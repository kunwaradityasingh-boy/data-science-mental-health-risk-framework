from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.dreaddit_features import (
    get_engineered_feature_columns,
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

    feature_columns = get_engineered_feature_columns(df)

    print("=" * 70)
    print("DREAddit FEATURE-LABEL ASSOCIATION AUDIT")
    print("=" * 70)

    print(f"\nEngineered features analysed: {len(feature_columns)}")

    correlations = []

    for feature in feature_columns:

        correlation = df[feature].corr(
            df["label"]
        )

        correlations.append(
            {
                "feature": feature,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
        )

    result = pd.DataFrame(correlations)

    result = result.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    print("\nTop 20 features by absolute Pearson correlation:")
    print()

    print(
        result[
            [
                "feature",
                "correlation",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("FEATURE-LABEL AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()