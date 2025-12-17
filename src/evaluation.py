import pandas as pd
from src.embeddings import load_embeddings
from src.recommender import recommend

def recall_at_10(pred, gold):
    return len(set(pred) & set(gold)) / len(gold)

df, model, embeddings = load_embeddings()

train = pd.read_excel("data/evaluation/train.xlsx")

scores = []

for _, row in train.iterrows():
    preds = recommend(row["query"], df, model, embeddings)
    score = recall_at_10(preds["url"].tolist(), row["relevant_urls"].split(","))
    scores.append(score)

print("Mean Recall@10:", sum(scores) / len(scores))
