import pickle
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer

EMBEDDINGS_PATH = "data/processed/shl_embeddings.pkl"
CATALOG_PATH = "data/processed/shl_catalog_clean.csv"

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")
df_catalog = pd.read_csv(CATALOG_PATH)

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings_data = pickle.load(f)

# Extract embeddings from dict (if saved as dict) or use directly (if saved as array)
if isinstance(embeddings_data, dict):
    corpus_embeddings = embeddings_data["embeddings"]
else:
    corpus_embeddings = embeddings_data

# Convert to tensor if it's not already
if not isinstance(corpus_embeddings, torch.Tensor):
    corpus_embeddings = torch.tensor(corpus_embeddings)


def recommend_assessments(query: str, top_k: int = 10):
    """
    Returns list of assessment URLs ranked by relevance
    """
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = scores.topk(k=top_k)

    recommendations = []
    for idx in top_results.indices:
        recommendations.append(df_catalog.iloc[int(idx)]["url"])

    return recommendations