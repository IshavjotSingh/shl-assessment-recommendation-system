import pandas as pd
from recommender import recommend_assessments

TEST_PATH = "data/test/test.xlsx"
OUTPUT_PATH = "data/outputs/test_predictions.csv"
TOP_K = 10

def generate_predictions():
    df_test = pd.read_excel(TEST_PATH, sheet_name="Test-Set")

    rows = []

    for _, row in df_test.iterrows():
        query = str(row["Query"]).strip()

        predicted_urls = recommend_assessments(query, top_k=TOP_K)

        for url in predicted_urls:
            rows.append({
                "Query": query,
                "Assessment_url": url
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Saved predictions → {OUTPUT_PATH}")
    print(f"Total rows: {len(out_df)}")

if __name__ == "__main__":
    generate_predictions()
