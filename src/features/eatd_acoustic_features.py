from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "acoustic_features.csv"
)

SAMPLE_RATE = 16000

POLARITIES = [
    "negative",
    "neutral",
    "positive",
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_audio_features(
    audio_path: Path,
) -> dict:

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if signal.size == 0:
        raise ValueError(
            "Audio contains zero samples."
        )

    duration = len(signal) / sr

    if duration <= 0:
        raise ValueError(
            "Audio duration is zero."
        )

    # --------------------------------------------------------
    # Basic signal statistics
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=signal
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        signal
    )[0]

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr,
    )[0]

    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sr,
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr,
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13,
    )

    delta_mfcc = librosa.feature.delta(
        mfcc
    )

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2,
    )

    features = {
        "duration_sec": duration,

        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_median": float(np.median(rms)),

        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_median": float(np.median(zcr)),

        "spectral_centroid_mean": float(
            np.mean(centroid)
        ),
        "spectral_centroid_std": float(
            np.std(centroid)
        ),

        "spectral_bandwidth_mean": float(
            np.mean(bandwidth)
        ),
        "spectral_bandwidth_std": float(
            np.std(bandwidth)
        ),

        "spectral_rolloff_mean": float(
            np.mean(rolloff)
        ),
        "spectral_rolloff_std": float(
            np.std(rolloff)
        ),
    }

    # --------------------------------------------------------
    # Spectral contrast
    # --------------------------------------------------------

    for i in range(
        contrast.shape[0]
    ):
        features[
            f"spectral_contrast_{i+1}_mean"
        ] = float(
            np.mean(contrast[i])
        )

        features[
            f"spectral_contrast_{i+1}_std"
        ] = float(
            np.std(contrast[i])
        )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    for i in range(
        chroma.shape[0]
    ):
        features[
            f"chroma_{i+1}_mean"
        ] = float(
            np.mean(chroma[i])
        )

        features[
            f"chroma_{i+1}_std"
        ] = float(
            np.std(chroma[i])
        )

    # --------------------------------------------------------
    # MFCC + Delta + Delta2
    # --------------------------------------------------------

    for i in range(
        mfcc.shape[0]
    ):
        coefficient = i + 1

        features[
            f"mfcc_{coefficient}_mean"
        ] = float(
            np.mean(mfcc[i])
        )

        features[
            f"mfcc_{coefficient}_std"
        ] = float(
            np.std(mfcc[i])
        )

        features[
            f"mfcc_{coefficient}_median"
        ] = float(
            np.median(mfcc[i])
        )

        features[
            f"delta_mfcc_{coefficient}_mean"
        ] = float(
            np.mean(delta_mfcc[i])
        )

        features[
            f"delta_mfcc_{coefficient}_std"
        ] = float(
            np.std(delta_mfcc[i])
        )

        features[
            f"delta2_mfcc_{coefficient}_mean"
        ] = float(
            np.mean(delta2_mfcc[i])
        )

        features[
            f"delta2_mfcc_{coefficient}_std"
        ] = float(
            np.std(delta2_mfcc[i])
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    if not ROOT.exists():
        raise FileNotFoundError(
            f"EATD dataset not found: {ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    skipped = []

    participant_count = 0

    for split_dir in sorted(
        ROOT.iterdir()
    ):

        if not split_dir.is_dir():
            continue

        split = split_dir.name

        for participant_dir in sorted(
            split_dir.iterdir()
        ):

            if not participant_dir.is_dir():
                continue

            label_file = (
                participant_dir
                / "label.txt"
            )

            new_label_file = (
                participant_dir
                / "new_label.txt"
            )

            if not label_file.exists():
                continue

            participant_count += 1

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

            participant_rows = []

            for polarity in POLARITIES:

                audio_file = (
                    participant_dir
                    / f"{polarity}_out.wav"
                )

                if not audio_file.exists():

                    skipped.append(
                        {
                            "split": split,
                            "participant":
                                participant_dir.name,
                            "polarity": polarity,
                            "reason":
                                "missing_processed_audio",
                        }
                    )

                    continue

                try:

                    features = (
                        extract_audio_features(
                            audio_file
                        )
                    )

                    features.update(
                        {
                            "split": split,
                            "participant":
                                participant_dir.name,
                            "polarity": polarity,
                            "label": label,
                            "new_label": new_label,
                        }
                    )

                    participant_rows.append(
                        features
                    )

                except Exception as exc:

                    skipped.append(
                        {
                            "split": split,
                            "participant":
                                participant_dir.name,
                            "polarity": polarity,
                            "reason":
                                str(exc),
                        }
                    )

            # ------------------------------------------------
            # Participant-level aggregation
            # ------------------------------------------------

            if participant_rows:

                response_df = pd.DataFrame(
                    participant_rows
                )

                numeric_columns = (
                    response_df
                    .select_dtypes(
                        include=np.number
                    )
                    .columns
                    .tolist()
                )

                # Do not aggregate labels
                numeric_feature_columns = [
                    column
                    for column in numeric_columns
                    if column
                    not in [
                        "label",
                        "new_label",
                    ]
                ]

                aggregated = (
                    response_df[
                        numeric_feature_columns
                    ]
                    .mean()
                    .to_dict()
                )

                aggregated.update(
                    {
                        "split": split,
                        "participant":
                            participant_dir.name,
                        "label": label,
                        "new_label": new_label,
                        "usable_responses":
                            len(participant_rows),
                    }
                )

                rows.append(
                    aggregated
                )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No acoustic features were extracted."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    metadata_columns = [
        "split",
        "participant",
        "label",
        "new_label",
        "usable_responses",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    numeric_features = df[
        feature_columns
    ]

    missing_values = int(
        numeric_features.isna()
        .sum()
        .sum()
    )

    infinite_values = int(
        np.isinf(
            numeric_features.to_numpy()
        ).sum()
    )

    duplicate_participants = (
        df[
            [
                "split",
                "participant",
            ]
        ]
        .duplicated()
        .sum()
    )

    invalid_response_count = int(
        (
            df["usable_responses"] < 1
        ).sum()
    )

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("EATD ACOUSTIC FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Participant records: {len(df)}"
    )

    print(
        f"Acoustic features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Usable response range: "
        f"{df['usable_responses'].min()} "
        f"to "
        f"{df['usable_responses'].max()}"
    )

    print(
        f"Missing feature values: "
        f"{missing_values}"
    )

    print(
        f"Infinite feature values: "
        f"{infinite_values}"
    )

    print(
        f"Duplicate split+participant rows: "
        f"{duplicate_participants}"
    )

    print(
        f"Participants with zero usable "
        f"responses: {invalid_response_count}"
    )

    print(
        f"Skipped audio records: "
        f"{len(skipped)}"
    )

    if skipped:
        print()
        print("Skipped records:")

        for item in skipped:
            print(
                f"  {item['split']}/"
                f"{item['participant']}/"
                f"{item['polarity']}: "
                f"{item['reason']}"
            )

    validation_passed = (
        len(df) > 0
        and missing_values == 0
        and infinite_values == 0
        and duplicate_participants == 0
        and invalid_response_count == 0
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()

    if validation_passed:
        print(
            "FEATURE EXTRACTION VALIDATION: PASSED"
        )
    else:
        print(
            "FEATURE EXTRACTION VALIDATION: "
            "REVIEW REQUIRED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()