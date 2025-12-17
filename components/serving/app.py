import os
from typing import Any, Dict, List, Optional

import boto3
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/tmp/model/model.joblib")
S3_BUCKET = os.getenv("MODEL_S3_BUCKET", "mlops-anime-data")
S3_KEY = os.getenv("MODEL_S3_KEY", "models/anime_recommender/approved/model.joblib")

DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))
MAX_K = int(os.getenv("MAX_K", "50"))

app = FastAPI(title="AnimeOps Recommender", version="0.2.0")

_model: Optional[Dict[str, Any]] = None
_titles_lower: Optional[List[str]] = None
_model_etag: Optional[str] = None


def _s3_client():
    return boto3.client("s3")


def _download_model() -> str:
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    s3 = _s3_client()
    head = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    etag = head.get("ETag", "").strip('"')
    s3.download_file(S3_BUCKET, S3_KEY, MODEL_LOCAL_PATH)
    return etag



def _load_model_from_disk() -> Dict[str, Any]:
    artifact = joblib.load(MODEL_LOCAL_PATH)
    # Minimal validation
    for k in ("titles", "vectorizer", "tfidf_matrix"):
        if k not in artifact:
            raise ValueError(f"Model artifact missing key: {k}")
    return artifact


def _build_title_index(titles: List[str]) -> List[str]:
    return [t.strip().lower() for t in titles]


def _find_title_index(query: str, titles_lower: List[str]) -> int:
    q = query.strip().lower()
    if not q:
        raise ValueError("title is empty")

    # exact match
    if q in titles_lower:
        return titles_lower.index(q)

    # contains match (simple fallback)
    for i, t in enumerate(titles_lower):
        if q in t:
            return i

    raise ValueError(f"Unknown title: {query}")


def _get_meta_row(meta: Dict[str, List[Any]], idx: int) -> Dict[str, Any]:
    if not meta:
        return {}
    out: Dict[str, Any] = {}
    # choose a small set of useful fields if present
    wanted = ["Score", "Popularity", "Rank", "Type", "Episodes", "Studios", "Genres"]
    for col in wanted:
        if col in meta and idx < len(meta[col]):
            val = meta[col][idx]
            # convert numpy scalars nicely
            if isinstance(val, (np.generic,)):
                val = val.item()
            if val is not None and str(val) != "nan":
                out[col.lower()] = val
    return out


def recommend_from_artifact(artifact: Dict[str, Any], title: str, k: int) -> List[Dict[str, Any]]:
    titles: List[str] = artifact["titles"]
    X = artifact["tfidf_matrix"]
    meta: Dict[str, List[Any]] = artifact.get("meta", {}) or {}

    global _titles_lower
    if _titles_lower is None or len(_titles_lower) != len(titles):
        _titles_lower = _build_title_index(titles)

    idx = _find_title_index(title, _titles_lower)

    # cosine sim between one vector and all items (efficient, no NxN)
    sims = cosine_similarity(X[idx], X).ravel()
    sims[idx] = -1.0  # exclude itself

    k = min(k, len(titles) - 1) if len(titles) > 1 else 0
    if k <= 0:
        return []

    top_idx = np.argpartition(-sims, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    recs: List[Dict[str, Any]] = []
    for j in top_idx:
        rec = {"title": titles[j], "similarity": float(sims[j])}
        rec.update(_get_meta_row(meta, j))
        recs.append(rec)
    return recs


@app.on_event("startup")
def startup_load():
    global _model, _titles_lower, _model_etag
    try:
        etag = _download_model()
        _model = _load_model_from_disk()
        _model_etag = etag
        _titles_lower = None
    except Exception as e:
        _model = None
        _titles_lower = None
        _model_etag = None
        print(f"Failed to load model: {e}")


@app.post("/reload")
def reload_model():
    """Manual reload without restart (handy during ops)."""
    global _model, _titles_lower, _model_etag
    try:
        etag = _download_model()
        _model = _load_model_from_disk()
        _model_etag = etag
        _titles_lower = None
        return {"status": "reloaded", "model_key": S3_KEY, "etag": _model_etag}
    except Exception as e:
        _model = None
        _titles_lower = None
        _model_etag = None
        raise HTTPException(status_code=500, detail=f"failed to reload model: {e}")


class RecommendResponse(BaseModel):
    query: str
    k: int
    recommendations: List[Dict[str, Any]]


@app.get("/healthz")
def healthz():
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_key": S3_KEY, "etag": _model_etag}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: str = Query(..., description="Anime title to base recommendations on"),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        recs = recommend_from_artifact(_model, title=title, k=k)
        return RecommendResponse(query=title, k=k, recommendations=recs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected error: {e}")
