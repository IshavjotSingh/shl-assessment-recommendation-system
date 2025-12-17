import pandas as pd
import os

INPUT_PATH = "data/raw/shl_catalog_raw.csv"
OUTPUT_PATH = "data/processed/shl_catalog_clean.csv"

def preprocess():
    df = pd.read_csv(INPUT_PATH)

    df = df.rename(columns={
        "assessment_name": "assessment_name",
        "url": "url",
        "remote_testing_support": "remote",
        "adaptive_irt": "adaptive",
        "test_type": "test_type"
    })

    df["combined_text"] = (
        df["assessment_name"] + " " +
        df["test_type"] + " " +
        df["remote"] + " " +
        df["adaptive"]
    )

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data → {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
