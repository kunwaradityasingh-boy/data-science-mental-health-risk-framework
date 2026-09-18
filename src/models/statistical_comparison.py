from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def compare_metric(
    df: pd.DataFrame,
    experiment_a: str,
    experiment_b: str,
    metric: str,
) -> None:
    """
    Compare paired metric values across the same validation seeds.
    """

    a = (
        df[df["experiment"] == experiment_a]
        .sort_values("seed")[metric]
        .reset_index(drop=True)
    )

    b = (
        df[df["experiment"] == experiment_b]
        .sort_values("seed")[metric]
        .reset_index(drop=True)
    )

    if len(a) != len(b):
        raise ValueError(
            "Experiments do not contain the same number of seeds."
        )

    differences = b - a

    print(f"\n{metric.upper()}")
    print("-" * 70)

    print(
        f"{experiment_a} mean: {a.mean():.4f}"
    )

    print(
        f"{experiment_b} mean: {b.mean():.4f}"
    )

    print(
        f"Mean paired difference: {differences.mean():+.4f}"
    )

    print(
        f"Difference std: {differences.std(ddof=1):.4f}"
    )

    # Paired t-test
    t_stat, t_pvalue = ttest_rel(
        b,
        a,
    )

    print(
        f"Paired t-test: "
        f"t={t_stat:.4f}, "
        f"p={t_pvalue:.4f}"
    )

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pvalue = wilcoxon(
            b,
            a,
            zero_method="wilcox",
        )

        print(
            f"Wilcoxon test: "
            f"W={w_stat:.4f}, "
            f"p={w_pvalue:.4f}"
        )

    except ValueError as error:
        print(
            f"Wilcoxon test unavailable: {error}"
        )


def main() -> None:

    base_dir = Path(__file__).resolve().parents[2]

    results_path = (
        base_dir
        / "reports"
        / "repeated_validation_results.csv"
    )

    df = pd.read_csv(
        results_path
    )

    print("=" * 75)
    print("STATISTICAL COMPARISON OF REPEATED VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"Seeds analysed: "
        f"{sorted(df['seed'].unique().tolist())}"
    )

    comparisons = [
        (
            "text_only",
            "text_plus_social",
        ),
        (
            "text_only",
            "text_plus_all_engineered",
        ),
    ]

    metrics = [
        "accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for experiment_a, experiment_b in comparisons:

        print()
        print("=" * 75)

        print(
            f"COMPARISON: "
            f"{experiment_a} vs {experiment_b}"
        )

        print("=" * 75)

        for metric in metrics:

            compare_metric(
                df,
                experiment_a,
                experiment_b,
                metric,
            )

    print()
    print("=" * 75)
    print("STATISTICAL COMPARISON COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()