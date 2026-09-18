from pathlib import Path

import pandas as pd
import requests


BASE_URL = "http://127.0.0.1:8000"
DATA_PATH = Path("data/raw/dreaddit/extracted/dreaddit-train.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    identifier_columns = {
        "id",
        "post_id",
        "subreddit",
        "text",
        "sentence_range",
        "label",
        "confidence",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in identifier_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    print(f"Total rows: {len(df)}")
    print(f"Engineered features: {len(feature_columns)}")

    assert len(feature_columns) == 109

    row = df.iloc[0]

    payload = {
        "text": str(row["text"]),
        "subject_id": f"research-subject-{row['id']}",
        "engineered_features": {
            column: float(row[column])
            for column in feature_columns
        },
        "available_modalities": ["text"],
        "missing_modalities": ["voice", "behavior"],
        "data_quality": "acceptable",
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        timeout=30,
    )

    print("\nHTTP status:", response.status_code)
    print("\nAPI response:")

    print(response.json())

    response.raise_for_status()


if __name__ == "__main__":
    main()