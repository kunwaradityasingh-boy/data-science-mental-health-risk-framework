from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "studentlife"
    / "extracted"
    / "dataset_rds"
    / "sensing"
    / "audio.Rds"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "studentlife"
    / "audio_features.csv"
)


def build_audio_features() -> pd.DataFrame:
    df = pd.read_pickle(INPUT_PATH) if INPUT_PATH.suffix == ".pkl" else None

    if df is None:
        import pyreadr

        result = pyreadr.read_r(str(INPUT_PATH))
        if not result:
            raise RuntimeError("Could not read audio.Rds")

        df = next(iter(result.values()))

    required_columns = {"timestamp", "audio_inference", "uid"}

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["timestamp", "audio_inference", "uid"]].copy()

    df["audio_inference"] = pd.to_numeric(
        df["audio_inference"], errors="coerce"
    )

    df = df.dropna(subset=["audio_inference", "uid"])

    # Verified StudentLife encoding:
    # 0 = Silence
    # 1 = Noise
    # 2 = Voices
    # 3 = Unknown

    grouped = []

    for uid, group in df.groupby("uid", sort=True):
        total = len(group)

        silence_count = int((group["audio_inference"] == 0).sum())
        noise_count = int((group["audio_inference"] == 1).sum())
        voice_count = int((group["audio_inference"] == 2).sum())
        unknown_count = int((group["audio_inference"] == 3).sum())

        grouped.append(
            {
                "uid": uid,
                "audio_silence_count": silence_count,
                "audio_noise_count": noise_count,
                "audio_voice_count": voice_count,
                "audio_unknown_count": unknown_count,
                "audio_observations": total,
                "audio_silence_pct": silence_count / total,
                "audio_noise_pct": noise_count / total,
                "audio_voice_pct": voice_count / total,
                "audio_unknown_pct": unknown_count / total,
                "audio_non_silence_pct": (
                    (noise_count + voice_count + unknown_count) / total
                ),
            }
        )

    features = pd.DataFrame(grouped)

    if features.empty:
        raise ValueError("No audio features were generated.")

    return features


def validate_audio_features(features: pd.DataFrame) -> None:
    if features["uid"].duplicated().any():
        raise AssertionError("Duplicate UIDs found.")

    numeric_columns = [
        column
        for column in features.columns
        if column != "uid"
    ]

    if features[numeric_columns].isna().any().any():
        raise AssertionError("Missing numeric values found.")

    if np.isinf(features[numeric_columns].to_numpy()).any():
        raise AssertionError("Infinite numeric values found.")

    count_columns = [
        "audio_silence_count",
        "audio_noise_count",
        "audio_voice_count",
        "audio_unknown_count",
    ]

    if not np.allclose(
        features[count_columns].sum(axis=1),
        features["audio_observations"],
    ):
        raise AssertionError("Audio count consistency failed.")

    pct_columns = [
        "audio_silence_pct",
        "audio_noise_pct",
        "audio_voice_pct",
        "audio_unknown_pct",
        "audio_non_silence_pct",
    ]

    for column in pct_columns:
        if not features[column].between(0, 1).all():
            raise AssertionError(
                f"{column} contains values outside [0, 1]."
            )

    pct_sum = (
        features["audio_silence_pct"]
        + features["audio_noise_pct"]
        + features["audio_voice_pct"]
        + features["audio_unknown_pct"]
    )

    if not np.allclose(pct_sum, 1.0):
        raise AssertionError("Audio percentage consistency failed.")

    expected_non_silence = (
        1.0 - features["audio_silence_pct"]
    )

    if not np.allclose(
        features["audio_non_silence_pct"],
        expected_non_silence,
    ):
        raise AssertionError(
            "Non-silence percentage consistency failed."
        )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = build_audio_features()

    validate_audio_features(features)

    features.to_csv(OUTPUT_PATH, index=False)

    print("Audio feature extraction complete.")
    print(f"Shape: {features.shape}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Unique UIDs: {features['uid'].nunique()}")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()