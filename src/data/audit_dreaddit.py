from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "dreaddit" / "extracted"

TRAIN_PATH = DATA_DIR / "dreaddit-train.csv"
TEST_PATH = DATA_DIR / "dreaddit-test.csv"


def inspect_dataset(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"{name} DATASET")
    print(f"{'=' * 60}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate IDs:")
    print(df["id"].duplicated().sum())

    print("\nDuplicate post IDs:")
    print(df["post_id"].duplicated().sum())

    print("\nUnique subreddits:")
    print(df["subreddit"].nunique())

    print("\nSubreddit distribution:")
    print(df["subreddit"].value_counts())

    print("\nText length statistics:")
    text_lengths = df["text"].astype(str).str.len()
    print(text_lengths.describe())

    return df


def main() -> None:
    train_df = inspect_dataset(TRAIN_PATH, "TRAIN")
    test_df = inspect_dataset(TEST_PATH, "TEST")

    print(f"\n{'=' * 60}")
    print("TRAIN / TEST OVERLAP")
    print(f"{'=' * 60}")

    train_ids = set(train_df["id"].astype(str))
    test_ids = set(test_df["id"].astype(str))

    train_post_ids = set(train_df["post_id"].astype(str))
    test_post_ids = set(test_df["post_id"].astype(str))

    train_text = set(train_df["text"].astype(str))
    test_text = set(test_df["text"].astype(str))

    print("Overlapping IDs:", len(train_ids & test_ids))
    print("Overlapping post IDs:", len(train_post_ids & test_post_ids))
    print("Overlapping exact text records:", len(train_text & test_text))


if __name__ == "__main__":
    main()