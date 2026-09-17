from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "dreaddit" / "extracted"

TRAIN_PATH = DATA_DIR / "dreaddit-train.csv"
TEST_PATH = DATA_DIR / "dreaddit-test.csv"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Exact text overlap
    train_text = set(train["text"].astype(str))
    test_text = set(test["text"].astype(str))

    overlapping_text = train_text & test_text

    print("=" * 70)
    print("EXACT TEXT OVERLAP")
    print("=" * 70)

    print("Number of overlapping texts:", len(overlapping_text))

    for text in sorted(overlapping_text):
        print("\n" + "-" * 70)
        print("TEXT:")
        print(text)

        print("\nTRAIN RECORDS:")
        print(
            train.loc[
                train["text"].astype(str) == text,
                ["id", "post_id", "subreddit", "sentence_range", "label"],
            ].to_string(index=False)
        )

        print("\nTEST RECORDS:")
        print(
            test.loc[
                test["text"].astype(str) == text,
                ["id", "post_id", "subreddit", "sentence_range", "label"],
            ].to_string(index=False)
        )

    # Post IDs within each split
    print("\n" + "=" * 70)
    print("DUPLICATE POST IDs")
    print("=" * 70)

    train_post_counts = train["post_id"].value_counts()
    test_post_counts = test["post_id"].value_counts()

    print("\nTRAIN:")
    print(train_post_counts[train_post_counts > 1].head(20))

    print("\nTEST:")
    print(test_post_counts[test_post_counts > 1].head(20))


if __name__ == "__main__":
    main()