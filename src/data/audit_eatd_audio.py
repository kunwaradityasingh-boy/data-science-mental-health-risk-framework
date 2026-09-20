from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "eatd"
    / "extracted"
    / "EATD-Corpus"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eatd_audio_metadata.csv"
)

POLARITIES = [
    "negative",
    "neutral",
    "positive",
]


def inspect_audio(path: Path):

    try:
        info = sf.info(path)

        duration = (
            info.frames / info.samplerate
            if info.samplerate > 0
            else 0.0
        )

        return {
            "status": "OK",
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_sec": duration,
            "subtype": info.subtype,
        }

    except Exception as exc:

        return {
            "status": f"ERROR: {exc}",
            "sample_rate": np.nan,
            "channels": np.nan,
            "frames": np.nan,
            "duration_sec": np.nan,
            "subtype": "",
        }


def main():

    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    rows = []

    for split_dir in sorted(ROOT.iterdir()):

        if not split_dir.is_dir():
            continue

        split = split_dir.name

        for participant_dir in sorted(
            split_dir.iterdir()
        ):

            if not participant_dir.is_dir():
                continue

            participant = participant_dir.name

            label_file = participant_dir / "label.txt"
            new_label_file = participant_dir / "new_label.txt"

            if not label_file.exists():
                continue

            label = float(
                label_file.read_text(
                    encoding="utf-8"
                ).strip()
            )

            new_label = float(
                new_label_file.read_text(
                    encoding="utf-8"
                ).strip()
            )

            for polarity in POLARITIES:

                raw_path = (
                    participant_dir
                    / f"{polarity}.wav"
                )

                processed_path = (
                    participant_dir
                    / f"{polarity}_out.wav"
                )

                raw = inspect_audio(raw_path)
                processed = inspect_audio(
                    processed_path
                )

                rows.append(
                    {
                        "split": split,
                        "participant": participant,
                        "polarity": polarity,
                        "label": label,
                        "new_label": new_label,

                        "raw_status": raw["status"],
                        "raw_sample_rate": raw["sample_rate"],
                        "raw_channels": raw["channels"],
                        "raw_duration_sec": raw["duration_sec"],

                        "processed_status": processed["status"],
                        "processed_sample_rate": processed["sample_rate"],
                        "processed_channels": processed["channels"],
                        "processed_duration_sec": processed[
                            "duration_sec"
                        ],
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No audio records found."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 65)
    print("EATD AUDIO QUALITY AUDIT")
    print("=" * 65)

    print(
        f"Audio records: {len(df)}"
    )

    print(
        f"Participants: "
        f"{df['participant'].nunique()}"
    )

    print(
        f"Expected records: "
        f"{df['participant'].nunique() * 3}"
    )

    print()
    print("Raw audio status:")
    print(
        df["raw_status"]
        .value_counts()
        .to_string()
    )

    print()
    print("Processed audio status:")
    print(
        df["processed_status"]
        .value_counts()
        .to_string()
    )

    print()
    print("Raw sample rates:")
    print(
        df["raw_sample_rate"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Processed sample rates:")
    print(
        df["processed_sample_rate"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Raw channels:")
    print(
        df["raw_channels"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Processed channels:")
    print(
        df["processed_channels"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Raw duration statistics (seconds):"
    )

    print(
        df["raw_duration_sec"]
        .describe()
        .to_string()
    )

    print()
    print(
        "Processed duration statistics (seconds):"
    )

    print(
        df["processed_duration_sec"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    raw_errors = (
        df["raw_status"] != "OK"
    ).sum()

    processed_errors = (
        df["processed_status"] != "OK"
    ).sum()

    zero_raw = (
        df["raw_duration_sec"] <= 0
    ).sum()

    zero_processed = (
        df["processed_duration_sec"] <= 0
    ).sum()

    invalid_duration = (
        df[
            [
                "raw_duration_sec",
                "processed_duration_sec",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    expected_records = (
        df["participant"].nunique() * 3
    )

    validation_passed = (
        len(df) == expected_records
        and raw_errors == 0
        and processed_errors == 0
        and zero_raw == 0
        and zero_processed == 0
        and invalid_duration == 0
    )

    print()
    print("=" * 65)

    print(
        f"Raw audio errors: {raw_errors}"
    )

    print(
        f"Processed audio errors: "
        f"{processed_errors}"
    )

    print(
        f"Zero-duration raw files: "
        f"{zero_raw}"
    )

    print(
        f"Zero-duration processed files: "
        f"{zero_processed}"
    )

    print(
        f"Invalid duration records: "
        f"{invalid_duration}"
    )

    print()

    if validation_passed:
        print(
            "AUDIO QUALITY VALIDATION: PASSED"
        )
    else:
        print(
            "AUDIO QUALITY VALIDATION: REVIEW REQUIRED"
        )

    print()
    print(
        f"Metadata saved to:\n{OUTPUT}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()