import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "train" / "trainset.xlsx"
TEST_PATH = BASE_DIR / "data" / "test" / "test.xlsx"

def load_data():
    train_df = pd.read_excel(TRAIN_PATH, sheet_name="Train-Set")
    test_df = pd.read_excel(TEST_PATH, sheet_name="Test-Set")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
