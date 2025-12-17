import pandas as pd

TRAIN_PATH = "data/train/trainset.xlsx"
TEST_PATH = "data/test/test.xlsx"

def load_data():
    train_df = pd.read_excel(TRAIN_PATH, sheet_name="Train-set")
    test_df = pd.read_excel(TEST_PATH, sheet_name="Test-set")
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data()
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print(train_df.head())
