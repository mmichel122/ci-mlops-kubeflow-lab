from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import boto3
import os

app = FastAPI(title="AnimeOps Recommender")

# --- Global Variables to hold model in memory ---
model_artifacts = {}

# --- Pydantic Models for Input/Output ---
class RecommendationRequest(BaseModel):
    anime_name: str
    top_k: int = 5

@app.on_event("startup")
def load_model():
    """
    On startup, download the model artifacts from S3.
    In production, you might want to pull from MLflow Model Registry instead.
    """
    print("Loading model artifacts...")
    bucket_name = os.getenv("BUCKET_NAME", "mlops-anime-data")
    
    # Simulate downloading (ensure these match what train.py saved)
    s3 = boto3.client('s3')
    
    # Helper to download and load
    def load_pkl_from_s3(key):
        local_path = f"/tmp/{key}"
        s3.download_file(bucket_name, f"model_artifacts/{key}", local_path)
        with open(local_path, "rb") as f:
            return pickle.load(f)

    # Load the 3 critical pieces
    global model_artifacts
    model_artifacts["similarity_matrix"] = load_pkl_from_s3("similarity_matrix.pkl")
    model_artifacts["metadata"] = pd.read_pickle(f"/tmp/metadata.pkl") # Downloaded similarly
    # (Simplified for brevity - assume metadata.pkl was downloaded to /tmp)
    
    print("Model loaded successfully!")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """
    Finds the anime in our database and returns top_k similar items.
    """
    df = model_artifacts["metadata"]
    sim_matrix = model_artifacts["similarity_matrix"]
    
    # 1. Find the index of the requested anime
    # Case-insensitive search
    match = df[df['English'].str.lower() == request.anime_name.lower()]
    
    if match.empty:
        # Fallback to Japanese title or Description search?
        # For now, 404
        raise HTTPException(status_code=404, detail="Anime not found in database")
    
    idx = match.index[0]
    
    # 2. Get Similarity Scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # 3. Sort by Score (High to Low)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 4. Get Top K (excluding itself at index 0)
    top_indices = [i[0] for i in sim_scores[1:request.top_k+1]]
    
    # 5. Return Results
    results = df.iloc[top_indices][['English', 'Genres', 'Score', 'Description']].to_dict(orient='records')
    return {"input": request.anime_name, "recommendations": results}