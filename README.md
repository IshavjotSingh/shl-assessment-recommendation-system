                            🧠 SHL Assessment Recommendation System (GenAI Task)

This project is an AI-powered Assessment Recommendation System built as part of the SHL GenAI Take-Home Assessment.
It helps hiring managers and recruiters find the most relevant SHL individual test solutions based on a natural language query, job description text, or URL.

The system combines:
Web scraping
Semantic search (Sentence-BERT)
LLM-based query understanding (Gemini)
Evaluation using Recall@K
REST API (FastAPI)
Interactive UI (Streamlit)

🚀 Key Features
Crawls and stores SHL Product Catalogue (Individual Test Solutions only)
Converts catalogue into semantic embeddings
Uses GenAI (Gemini) for intelligent query understanding
Returns 5–10 relevant assessments
Provides:
REST API
Web UI
Supports evaluation using labelled train data
Generates predictions for unlabelled test set


                                        Project Architecture

Intelligent_Assessment_Recommendation_System/
│
├── src/
│   ├── ingestion.py                  # Scrape SHL catalogue
│   ├── preprocessing.py              # Clean & prepare data
│   ├── embeddings.py                 # Build Sentence-BERT embeddings
│   ├── recommender.py                # Core recommendation logic
│   ├── data_loader.py                # Load train/test datasets
│   ├── evaluate.py                   # Mean Recall@K evaluation
│   ├── generate_test_predictions.py  # Generate test-set predictions
│   ├── api.py / main.py               # FastAPI backend
│
├── app.py                             # Streamlit frontend
├── query_functions.py                 # LLM + retrieval pipeline
│
├── data/
│   ├── raw/                           # Scraped catalogue
│   ├── processed/                     # Cleaned data + embeddings
│   ├── train/                         # Train-set (xlsx)
│   ├── test/                          # Test-set (xlsx)
│   └── outputs/                       # Predictions CSV
│
├── evaluation/                        # Evaluation artifacts
├── notebooks/                         # Experiments & exploration
│
├── requirements.txt
├── .env                               # API keys (not committed)
└── README.md


Environment Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Set Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

Step 1: Scrape SHL Product Catalogue
Scrapes Individual Test Solutions and filters out Pre-packaged Job Solutions.
python src/ingestion.py

Output:
data/raw/shl_catalog_raw.csv
data/raw/shl_catalog_filtered_out.csv

✔️ Meets requirement of crawling SHL catalogue
✔️ Uses Playwright to handle dynamic React UI

Step 2: Preprocess Catalogue
python src/preprocessing.py
Output:
data/processed/shl_catalog_clean.csv

Cleans columns
Builds combined text for embeddings

🔎 Step 3: Build Embeddings
python src/embeddings.py
Output:
data/processed/shl_embeddings.pkl

Uses Sentence-BERT (all-MiniLM-L6-v2)
Creates vector representations for semantic search

🤖 Step 4: Recommendation Pipeline
The recommendation logic:
User query → LLM (Gemini) → structured intent
Intent → semantic similarity search
Top-K assessments returned

Implemented in:

src/recommender.py
query_functions.py

📊 Step 5: Evaluation (Mean Recall@10)
python src/evaluate.py


Metric:

Mean Recall@10 on labelled train set
⚠️ Note: Due to small labelled data and catalogue drift, recall may be low.
The evaluation pipeline is implemented correctly as required.

🧪 Step 6: Generate Test Predictions
python src/generate_test_predictions.py

Output:
data/outputs/test_predictions.csv

Format:

Query,Assessment_url
Query 1,https://www.shl.com/...
Query 1,https://www.shl.com/...
...
✔️ Matches Appendix-3 submission format

🌐 FastAPI Backend
Run API
uvicorn main:app --reload

Endpoints
Health Check
GET /health


Response:

{ "status": "healthy" }

Recommendation
POST /recommend


Request:

{ "query": "Need a Java developer with collaboration skills" }


Response:

{
  "recommended_assessments": [
    {
      "assessment_name": "...",
      "url": "...",
      "adaptive_support": "...",
      "description": "...",
      "duration": 40,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"],
      "skills": ["Java"]
    }
  ]
}


API Docs:

http://127.0.0.1:8000/docs

🖥️ Streamlit Frontend

Run UI:

streamlit run app.py


Features:

Text input for queries

Clickable assessment links

Tabular output

📈 Technology Stack

Python

FastAPI

Streamlit

Sentence-Transformers

Google Gemini API

Playwright

Pandas / NumPy

PyTorch

✅ Submission Checklist

✔️ Scraped SHL catalogue
✔️ API endpoint live
✔️ Web UI available
✔️ Evaluation implemented
✔️ Test predictions CSV generated
✔️ Code pushed to GitHub

                                                      👤 Author

                                                      Ishavjot Singh
                                                      GenAI / Data Engineering Enthusiast