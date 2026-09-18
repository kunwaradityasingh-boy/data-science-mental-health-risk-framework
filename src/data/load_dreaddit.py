from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "dreaddit"
    / "extracted"
    / "dreaddit-train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "dreaddit"
    / "extracted"
    / "dreaddit-test.csv"
)


REQUIRED_COLUMNS = {
    "post_id",
    "text",
    "label",
}


def load_dreaddit_csv(path: Path) -> pd.DataFrame:
    """Load and perform basic structural validation on a Dreaddit CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def load_dreaddit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preserved Dreaddit train and test datasets."""

    train_df = load_dreaddit_csv(TRAIN_PATH)
    test_df = load_dreaddit_csv(TEST_PATH)

    return train_df, test_df


def print_dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a concise dataset summary."""

    print("=" * 60)
    print("DREAddit DATASET LOADER")
    print("=" * 60)

    print(f"Train rows    : {len(train_df)}")
    print(f"Train columns : {len(train_df.columns)}")
    print(f"Test rows     : {len(test_df)}")
    print(f"Test columns  : {len(test_df.columns)}")

    print("\nRequired columns:")
    for column in sorted(REQUIRED_COLUMNS):
        print(f"  [OK] {column}")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts().sort_index())

    print("\nPost statistics:")
    print(f"  Train unique posts: {train_df['post_id'].nunique()}")
    print(f"  Test unique posts : {test_df['post_id'].nunique()}")

    print("=" * 60)


if __name__ == "__main__":
    train, test = load_dreaddit()
    print_dataset_summary(train, test)