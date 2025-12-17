import json
import numpy as np
import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ---------------- LOAD DATA ----------------
CATALOG_PATH = "data/processed/shl_catalog_clean.csv"

catalog_df = pd.read_csv(CATALOG_PATH)

corpus = catalog_df["combined_text"].fillna("").tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# ---------------- LLM SETUP ----------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- HELPERS ----------------
def extract_features_with_llm(user_query: str) -> str:
    prompt = f"""
Extract key hiring intent from the query below.
Return a concise sentence with skills, role, constraints.

Query:
{user_query}
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def find_assessments(query: str, k: int = 10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(k, len(scores))
    top_results = torch.topk(scores, k=top_k)

    rows = []
    for idx, score in zip(top_results.indices, top_results.values):
        row = catalog_df.iloc[int(idx)]
        rows.append({
            "Assessment Name": row["assessment_name"],
            "URL": row["url"],
            "Remote Testing Support": row["remote"],
            "Adaptive/IRT": row["adaptive"],
            "Test Type": row["test_type"],
            "Description": row["combined_text"],
            "Score": float(score)
        })
    return rows

def query_handling_using_LLM_updated(query: str):
    refined_query = extract_features_with_llm(query)
    results = find_assessments(refined_query, k=10)
    return pd.DataFrame(results)
