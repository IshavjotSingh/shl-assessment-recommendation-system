import pandas as pd
from recommender import recommend_assessments
from urllib.parse import urlparse, urlunparse
import re

TRAIN_PATH = "data/train/trainset.xlsx"
K = 10


def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison:
    - Remove trailing slashes
    - Convert to lowercase
    - Remove query parameters and fragments
    - Normalize http/https
    """
    if not url or pd.isna(url):
        return ""
    
    url = str(url).strip()
    if not url:
        return ""
    
    # Parse URL
    parsed = urlparse(url.lower())
    
    # Reconstruct without query/fragment, normalize scheme
    normalized = urlunparse((
        parsed.scheme or "https",  # Default to https
        parsed.netloc,
        parsed.path.rstrip("/"),  # Remove trailing slash
        "",  # params
        "",  # query
        ""   # fragment
    ))
    
    return normalized


def extract_url_slug(url: str) -> str:
    """Extract the last meaningful part of URL path for fuzzy matching"""
    if not url:
        return ""
    parsed = urlparse(str(url).lower())
    path = parsed.path.rstrip("/")
    # Get last segment
    parts = [p for p in path.split("/") if p]
    return parts[-1] if parts else ""


def recall_at_k(predicted, relevant, k):
    predicted_k = predicted[:k]
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0.0

    hits = sum(1 for p in predicted_k if p in relevant_set)
    return hits / len(relevant_set)


def evaluate_mean_recall():
    train_df = pd.read_excel(TRAIN_PATH)

    print("Available columns:", train_df.columns.tolist())
    print("Rows:", len(train_df))

    query_col = "Query"
    url_col = "Assessment_url"

    recalls = []
    debug_samples = []

    for idx, row in train_df.iterrows():
        query = str(row[query_col]).strip()
        
        # Parse and normalize relevant URLs
        relevant_urls_raw = str(row[url_col]).split(",")
        relevant_urls = [normalize_url(u) for u in relevant_urls_raw if u.strip()]
        
        if not relevant_urls:
            continue

        # Get predictions
        predicted_urls_raw = recommend_assessments(query, top_k=K)
        predicted_urls = [normalize_url(u) for u in predicted_urls_raw]

        # Store first 3 examples for detailed debugging
        if len(debug_samples) < 3:
            # Also try fuzzy matching by slug
            relevant_slugs = [extract_url_slug(u) for u in relevant_urls]
            predicted_slugs = [extract_url_slug(u) for u in predicted_urls]
            slug_hits = sum(1 for p_slug in predicted_slugs if p_slug in set(relevant_slugs))
            
            debug_samples.append({
                "query": query,
                "relevant_raw": relevant_urls_raw[:3],
                "relevant_normalized": relevant_urls[:3],
                "predicted_raw": predicted_urls_raw[:3],
                "predicted_normalized": predicted_urls[:3],
                "exact_hits": sum(1 for p in predicted_urls if p in set(relevant_urls)),
                "slug_hits": slug_hits
            })

        r = recall_at_k(predicted_urls, relevant_urls, K)
        recalls.append(r)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(train_df)}")

    if not recalls:
        print("❌ No valid rows for evaluation")
        return

    # Print detailed debug samples
    print("\n" + "="*80)
    print("🔍 DETAILED DEBUG: First 3 examples")
    print("="*80)
    for i, sample in enumerate(debug_samples, 1):
        print(f"\n📋 Example {i}:")
        print(f"  Query: {sample['query']}")
        print(f"\n  📌 Relevant URLs (RAW from train data):")
        for j, url in enumerate(sample['relevant_raw'][:3], 1):
            print(f"    {j}. {url}")
        print(f"\n  📌 Relevant URLs (NORMALIZED):")
        for j, url in enumerate(sample['relevant_normalized'][:3], 1):
            print(f"    {j}. {url}")
        print(f"\n  🤖 Predicted URLs (RAW from recommender):")
        for j, url in enumerate(sample['predicted_raw'][:3], 1):
            print(f"    {j}. {url}")
        print(f"\n  🤖 Predicted URLs (NORMALIZED):")
        for j, url in enumerate(sample['predicted_normalized'][:3], 1):
            print(f"    {j}. {url}")
        print(f"\n  ✅ Exact URL matches in top-{K}: {sample['exact_hits']}")
        print(f"  🔗 Slug matches in top-{K}: {sample['slug_hits']}")

    mean_recall = sum(recalls) / len(recalls)
    print(f"\n" + "="*80)
    print(f"✅ Mean Recall@{K}: {mean_recall:.4f}")
    print(f"Evaluated on {len(recalls)} queries")
    
    # Show recall distribution
    recall_counts = {}
    for r in recalls:
        bucket = f"{int(r * 10) * 10}%"
        recall_counts[bucket] = recall_counts.get(bucket, 0) + 1
    print(f"\nRecall distribution: {recall_counts}")


if __name__ == "__main__":
    evaluate_mean_recall()