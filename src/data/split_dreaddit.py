from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split Dreaddit at post level so segments from the same post
    cannot appear in both training and validation sets.
    """

    if "post_id" not in df.columns:
        raise ValueError("Dataset must contain 'post_id'.")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain 'text'.")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label'.")

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


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Dreaddit training file not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    train_df, validation_df = create_post_aware_split(df)

    train_posts = set(train_df["post_id"])
    validation_posts = set(validation_df["post_id"])

    overlapping_posts = train_posts & validation_posts

    print("=" * 60)
    print("DREAddit POST-AWARE VALIDATION SPLIT")
    print("=" * 60)

    print(f"Original rows       : {len(df)}")
    print(f"Training rows       : {len(train_df)}")
    print(f"Validation rows     : {len(validation_df)}")

    print()
    print(f"Training posts      : {train_df['post_id'].nunique()}")
    print(f"Validation posts    : {validation_df['post_id'].nunique()}")

    print()
    print(f"Post overlap        : {len(overlapping_posts)}")

    print()
    print("Training label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print()
    print("Validation label distribution:")
    print(validation_df["label"].value_counts().sort_index())

    print("=" * 60)

    if overlapping_posts:
        raise RuntimeError(
            "Leakage detected: some post_ids appear in both splits."
        )

    print("[OK] No post_id overlap between training and validation.")


if __name__ == "__main__":
    main()