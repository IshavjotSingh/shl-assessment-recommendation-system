from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from query_functions import query_handling_using_LLM_updated  
from sentence_transformers import SentenceTransformer
import os
import torch
import google.generativeai as genai
from dotenv import load_dotenv

app = FastAPI()

# Global objects to be initialized on startup
model = None
gemini_model = None
catalog_df = None
corpus = None
corpus_embeddings = None


@app.on_event("startup")
def startup_event():
    global model, gemini_model, catalog_df, corpus, corpus_embeddings

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    print("🚀 Loading models and data...")

    # Load Sentence Transformer model FIRST
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load Gemini
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    # Load and process catalog data
    # Use the processed catalog (check which file exists)
    catalog_path = "data/processed/shl_catalog_clean.csv"
    if not os.path.exists(catalog_path):
        # Fallback to old format if processed doesn't exist
        catalog_path = "SHL_catalog.csv"
    
    catalog_df = pd.read_csv(catalog_path)

    # Check which columns exist and create combined text accordingly
    if "combined_text" in catalog_df.columns:
        # Already has combined_text
        corpus = catalog_df["combined_text"].tolist()
    else:
        # Need to create combined text from individual columns
        def combine_row(row):
            parts = []
            if "Assessment Name" in catalog_df.columns:
                parts.append(str(row.get("Assessment Name", "")))
            elif "assessment_name" in catalog_df.columns:
                parts.append(str(row.get("assessment_name", "")))
            
            if "Duration" in catalog_df.columns:
                parts.append(str(row.get("Duration", "")))
            if "Remote Testing Support" in catalog_df.columns:
                parts.append(str(row.get("Remote Testing Support", "")))
            elif "remote_testing_support" in catalog_df.columns:
                parts.append(str(row.get("remote_testing_support", "")))
            if "Adaptive/IRT" in catalog_df.columns:
                parts.append(str(row.get("Adaptive/IRT", "")))
            elif "adaptive_irt" in catalog_df.columns:
                parts.append(str(row.get("adaptive_irt", "")))
            if "Test Type" in catalog_df.columns:
                parts.append(str(row.get("Test Type", "")))
            elif "test_type" in catalog_df.columns:
                parts.append(str(row.get("test_type", "")))
            if "Skills" in catalog_df.columns:
                parts.append(str(row.get("Skills", "")))
            if "Description" in catalog_df.columns:
                parts.append(str(row.get("Description", "")))
            return ' '.join(parts)

        catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
        corpus = catalog_df['combined'].tolist()

    # Now create embeddings AFTER model is loaded
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    print("✅ Startup complete.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/info")
def info():
    return {
        "message": "SHL Assessment Recommendation API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }

# Request body
class QueryRequest(BaseModel):
    query: str

# Response model
class Assessment(BaseModel):
    assessment_name: str
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]
    skills: List[str]  

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(request: QueryRequest):
    if model is None or catalog_df is None or corpus_embeddings is None:
        raise HTTPException(status_code=503, detail="Service not ready. Models still loading.")
    
    try:
        df: pd.DataFrame = query_handling_using_LLM_updated(
            request.query,
            model=model,
            gemini_model=gemini_model,
            catalog_df=catalog_df,
            corpus=corpus,
            corpus_embeddings=corpus_embeddings
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="No assessments found.")

        results = []

        for _, row in df.iterrows():
            # Handle both old and new column names
            assessment_name = row.get("Assessment Name") or row.get("assessment_name", "")
            url = row.get("URL") or row.get("url", "")
            adaptive = row.get("Adaptive/IRT") or row.get("adaptive_irt", "")
            description = row.get("Description") or row.get("description", "")
            duration = row.get("Duration") or row.get("duration", 0)
            remote = row.get("Remote Testing Support") or row.get("remote_testing_support", "")
            test_type = row.get("Test Type") or row.get("test_type", "")
            skills = row.get("Skills") or row.get("skills", "")
            
            results.append({
                "assessment_name": assessment_name,
                "url": url,
                "adaptive_support": adaptive,
                "description": description,
                "duration": int(duration) if duration else 0,
                "remote_support": remote,
                "test_type": test_type if isinstance(test_type, list) else [test_type] if test_type else [],
                "skills": skills if isinstance(skills, list) else [skill.strip() for skill in str(skills).split(",")] if skills else []
            })

        return {"recommended_assessments": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))