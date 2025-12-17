import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer

# Paths
CLEAN_DATA_PATH = "data/processed/shl_catalog_clean.csv"
EMBEDDINGS_PATH = "data/processed/shl_embeddings.pkl"

def build_embeddings():
    print("🔹 Loading cleaned SHL catalog...")
    df = pd.read_csv(CLEAN_DATA_PATH)

    # ✅ Use correct column
    if "combined_text" not in df.columns:
        raise ValueError("combined_text column not found in cleaned data")

    texts = df["combined_text"].fillna("").tolist()

    print("🔹 Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔹 Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs("data/processed", exist_ok=True)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "metadata": df[[
                    "id",
                    "assessment_name",
                    "url",
                    "remote",
                    "adaptive",
                    "test_type"
                ]]
            },
            f
        )

    print(f"✅ Saved embeddings → {EMBEDDINGS_PATH}")
    print(f"✅ Total embeddings created: {len(embeddings)}")

if __name__ == "__main__":
    build_embeddings()
