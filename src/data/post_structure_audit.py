from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "dreaddit" / "extracted"

TRAIN_PATH = DATA_DIR / "dreaddit-train.csv"
TEST_PATH = DATA_DIR / "dreaddit-test.csv"


def audit_post_structure(df: pd.DataFrame, name: str) -> None:
    print("\n" + "=" * 70)
    print(f"{name} POST STRUCTURE")
    print("=" * 70)

    total_segments = len(df)
    unique_posts = df["post_id"].nunique()

    print(f"Total segments: {total_segments}")
    print(f"Unique posts: {unique_posts}")
    print(
        f"Segments per post: "
        f"{total_segments / unique_posts:.2f}"
    )

    counts = df["post_id"].value_counts()

    print("\nSegments-per-post distribution:")
    print(counts.describe())

    print("\nPosts with more than one segment:")
    print((counts > 1).sum())

    print("\nPosts with exactly one segment:")
    print((counts == 1).sum())

    print("\nMaximum segments from one post:")
    print(counts.max())

    print("\nTop 10 posts by segment count:")
    print(counts.head(10).to_string())


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    audit_post_structure(train, "TRAIN")
    audit_post_structure(test, "TEST")

    train_posts = set(train["post_id"].astype(str))
    test_posts = set(test["post_id"].astype(str))

    print("\n" + "=" * 70)
    print("CROSS-SPLIT POST OVERLAP")
    print("=" * 70)

    overlap = train_posts & test_posts

    print("Unique train posts:", len(train_posts))
    print("Unique test posts:", len(test_posts))
    print("Overlapping post IDs:", len(overlap))

    if overlap:
        print("\nOverlapping post IDs:")
        print(sorted(overlap))


if __name__ == "__main__":
    main()