from pathlib import Path

import pandas as pd


# ============================================================
# EATD TEXT DATA PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EATD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eatd"
    / "extracted"
    / "EATD-Corpus"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_FILES = [
    "negative.txt",
    "neutral.txt",
    "positive.txt",
]

SDS_THRESHOLD = 52.0


# ============================================================
# VALIDATE DATASET
# ============================================================

if not EATD_ROOT.exists():
    raise FileNotFoundError(
        f"EATD dataset not found:\n{EATD_ROOT}"
    )


print("=" * 70)
print("EATD TEXT DATA PREPARATION")
print("=" * 70)


# ============================================================
# PROCESS SPLIT
# ============================================================

records = []

for split in ["train", "validation"]:

    split_dir = EATD_ROOT / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing split directory:\n{split_dir}"
        )

    participant_dirs = [
        path
        for path in split_dir.iterdir()
        if path.is_dir()
    ]

    print()
    print(
        f"{split}: "
        f"{len(participant_dirs)} participants"
    )

    for participant_dir in sorted(
        participant_dirs,
        key=lambda x: x.name,
    ):

        participant = participant_dir.name

        label_file = (
            participant_dir
            / "label.txt"
        )

        new_label_file = (
            participant_dir
            / "new_label.txt"
        )

        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing label.txt:\n"
                f"{label_file}"
            )

        if not new_label_file.exists():
            raise FileNotFoundError(
                f"Missing new_label.txt:\n"
                f"{new_label_file}"
            )

        label = float(
            label_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        )

        new_label = float(
            new_label_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        )

        text_parts = []

        missing_text_files = []

        for text_file in TEXT_FILES:

            file_path = (
                participant_dir
                / text_file
            )

            if not file_path.exists():

                missing_text_files.append(
                    text_file
                )

                continue

            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if text:
                text_parts.append(
                    text
                )

        if missing_text_files:
            raise ValueError(
                f"Participant {participant} "
                f"in {split} is missing: "
                f"{missing_text_files}"
            )

        if len(text_parts) != 3:
            raise ValueError(
                f"Participant {participant} "
                f"in {split} does not have "
                f"three usable text responses."
            )

        combined_text = "\n".join(
            text_parts
        )

        target = int(
            new_label > SDS_THRESHOLD
        )

        records.append(
            {
                "split": split,
                "participant": participant,
                "label": label,
                "new_label": new_label,
                "target": target,
                "text": combined_text,
                "text_length": len(
                    combined_text
                ),
                "text_characters": len(
                    combined_text
                ),
            }
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

text_df = pd.DataFrame(
    records
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("TEXT DATA VALIDATION")
print("=" * 70)

print(
    f"Participant records: "
    f"{len(text_df)}"
)

print(
    f"Train records: "
    f"{(text_df['split'] == 'train').sum()}"
)

print(
    f"Validation records: "
    f"{(text_df['split'] == 'validation').sum()}"
)

print(
    f"Missing text values: "
    f"{text_df['text'].isna().sum()}"
)

print(
    f"Empty text values: "
    f"{(text_df['text'].str.strip() == '').sum()}"
)

print(
    f"Duplicate split+participant rows: "
    f"{text_df.duplicated(['split', 'participant']).sum()}"
)

print()
print("Target distribution:")

print(
    text_df["target"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "non-depressed",
            1: "depressed",
        }
    )
)

print()
print("Text length statistics:")

print(
    text_df["text_length"].describe()
)


# ============================================================
# TARGET CONSISTENCY CHECK
# ============================================================

expected_target = (
    text_df["new_label"]
    > SDS_THRESHOLD
).astype(int)

if not (
    text_df["target"]
    == expected_target
).all():

    raise ValueError(
        "Target construction inconsistency."
    )


# ============================================================
# LABEL TRANSFORMATION CHECK
# ============================================================

label_relation = np_allclose = (
    abs(
        text_df["new_label"]
        - text_df["label"] * 1.25
    )
    < 1e-6
)

if not label_relation.all():

    raise ValueError(
        "label/new_label relationship failed."
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    OUTPUT_DIR
    / "eatd_text_data.csv"
)

text_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

print(
    f"Output: {output_file}"
)

print()
print(
    "EATD TEXT DATA PREPARATION: PASSED"
)

print("=" * 70)