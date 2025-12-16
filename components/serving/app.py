\
import os
from typing import List, Dict, Any, Optional

import boto3
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/tmp/model/model.joblib")
S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "mlops-anime-data")
S3_KEY = os.getenv("MODEL_S3_KEY", "models/anime_recommender/model.joblib")

app = FastAPI(title="AnimeOps Recommender", version="0.1.0")
_model = None


def _s3_client():
    return boto3.client("s3")


def _download_model():
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    _s3_client().download_file(S3_BUCKET, S3_KEY, MODEL_LOCAL_PATH)


@app.on_event("startup")
def load_model():
    global _model
    try:
        _download_model()
        _model = joblib.load(MODEL_LOCAL_PATH)
    except Exception as e:
        # Keep the service up; requests will return 503 until model loads.
        _model = None
        app.logger = getattr(app, "logger", None)
        if app.logger:
            app.logger.error(f"Failed to load model: {e}")


class RecommendResponse(BaseModel):
    query: str
    k: int
    recommendations: List[Dict[str, Any]]


@app.get("/healthz")
def healthz():
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: str = Query(..., description="Anime title to base recommendations on"),
    k: int = Query(10, ge=1, le=50),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        recs = _model.recommend(title=title, k=k)
        return RecommendResponse(query=title, k=k, recommendations=recs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected error: {e}")
