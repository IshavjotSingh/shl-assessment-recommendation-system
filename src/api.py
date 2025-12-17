from fastapi import FastAPI
from pydantic import BaseModel
from src.embeddings import load_embeddings
from src.recommender import recommend

app = FastAPI()

df, model, embeddings = load_embeddings()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.get("/")
def root():
    return {"message": "API running. Go to /docs or /health"}

@app.post("/recommend")
def recommend_api(req: QueryRequest):
    recs = recommend(req.query, df, model, embeddings)
    return {
        "recommendations": recs.to_dict(orient="records")
    }
