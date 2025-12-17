import math
import os
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

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

app = FastAPI(title="AnimeOps Recommender", version="0.3.0")

_model_lock = RLock()

_model: Optional[Dict[str, Any]] = None
_titles_lower: Optional[List[str]] = None
_title_to_idx: Optional[Dict[str, int]] = None
_model_etag: Optional[str] = None

_s3_client_cached = None


def _s3_client():
    global _s3_client_cached
    if _s3_client_cached is None:
        _s3_client_cached = boto3.client("s3")
    return _s3_client_cached


def _head_model_etag() -> str:
    s3 = _s3_client()
    head = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    return head.get("ETag", "").strip('"')


def _download_model() -> str:
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    s3 = _s3_client()
    etag = _head_model_etag()
    s3.download_file(S3_BUCKET, S3_KEY, MODEL_LOCAL_PATH)
    return etag


def _load_model_from_disk() -> Dict[str, Any]:
    artifact = joblib.load(MODEL_LOCAL_PATH)
    for k in ("titles", "vectorizer", "tfidf_matrix"):
        if k not in artifact:
            raise ValueError(f"Model artifact missing key: {k}")
    return artifact


def _rebuild_title_index(titles: List[str]) -> Tuple[List[str], Dict[str, int]]:
    titles_lower = [t.strip().lower() for t in titles]
    title_to_idx = {t: i for i, t in enumerate(titles_lower)}
    return titles_lower, title_to_idx


def _find_title_index(query: str, titles_lower: List[str], title_to_idx: Dict[str, int]) -> int:
    q = query.strip().lower()
    if not q:
        raise ValueError("title is empty")

    # Exact match fast path
    if q in title_to_idx:
        return title_to_idx[q]

    # Substring fallback (first match)
    for i, t in enumerate(titles_lower):
        if q in t:
            return i

    raise ValueError(f"Unknown title: {query}")


def _is_nan(v: Any) -> bool:
    if isinstance(v, (np.generic,)):
        v = v.item()
    return isinstance(v, float) and math.isnan(v)


def _get_meta_row(meta: Dict[str, List[Any]], idx: int) -> Dict[str, Any]:
    if not meta:
        return {}

    out: Dict[str, Any] = {}
    wanted = ["Score", "Popularity", "Rank", "Type", "Episodes", "Studios", "Genres"]

    for col in wanted:
        if col not in meta or idx >= len(meta[col]):
            continue

        val = meta[col][idx]
        if isinstance(val, (np.generic,)):
            val = val.item()

        if val is None or _is_nan(val):
            continue

        out[col.lower()] = val

    return out


def recommend_from_artifact(artifact: Dict[str, Any], title: str, k: int) -> List[Dict[str, Any]]:
    titles: List[str] = artifact["titles"]
    X = artifact["tfidf_matrix"]
    meta: Dict[str, List[Any]] = artifact.get("meta", {}) or {}

    global _titles_lower, _title_to_idx
    if _titles_lower is None or _title_to_idx is None or len(_titles_lower) != len(titles):
        _titles_lower, _title_to_idx = _rebuild_title_index(titles)

    idx = _find_title_index(title, _titles_lower, _title_to_idx)

    sims = cosine_similarity(X[idx], X).ravel()
    sims[idx] = -1.0

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


def _load_model(force_download: bool) -> None:
    global _model, _titles_lower, _title_to_idx, _model_etag

    current_etag = _head_model_etag()
    if (not force_download) and _model is not None and _model_etag == current_etag:
        # Model already loaded and unchanged
        return

    etag = _download_model()
    artifact = _load_model_from_disk()

    _model = artifact
    _model_etag = etag
    _titles_lower = None
    _title_to_idx = None

    print(f"Loaded model s3://{S3_BUCKET}/{S3_KEY} etag={etag}")


@app.on_event("startup")
def startup_load():
    global _model, _titles_lower, _title_to_idx, _model_etag
    try:
        with _model_lock:
            _load_model(force_download=True)
    except Exception as e:
        _model = None
        _titles_lower = None
        _title_to_idx = None
        _model_etag = None
        print(f"Failed to load model: {e}")


class RecommendResponse(BaseModel):
    query: str
    k: int
    recommendations: List[Dict[str, Any]]


@app.get("/healthz")
def healthz():
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_key": S3_KEY, "etag": _model_etag}


@app.post("/reload")
def reload_model():
    global _model, _titles_lower, _title_to_idx, _model_etag
    try:
        with _model_lock:
            _load_model(force_download=False)
        return {"status": "reloaded", "model_key": S3_KEY, "etag": _model_etag}
    except Exception as e:
        _model = None
        _titles_lower = None
        _title_to_idx = None
        _model_etag = None
        raise HTTPException(status_code=500, detail=f"failed to reload model: {e}")


@app.get("/titles")
def titles(q: str = Query("", description="Substring search"), limit: int = Query(20, ge=1, le=100)):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    with _model_lock:
        artifact = _model
        if artifact is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        titles_list: List[str] = artifact["titles"]

        global _titles_lower, _title_to_idx
        if _titles_lower is None or _title_to_idx is None or len(_titles_lower) != len(titles_list):
            _titles_lower, _title_to_idx = _rebuild_title_index(titles_list)

        qq = q.strip().lower()
        if not qq:
            return {"q": q, "results": titles_list[:limit]}

        out: List[str] = []
        for i, t in enumerate(_titles_lower):
            if qq in t:
                out.append(titles_list[i])
                if len(out) >= limit:
                    break

        return {"q": q, "results": out}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: str = Query(..., description="Anime title to base recommendations on"),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        with _model_lock:
            artifact = _model
            if artifact is None:
                raise HTTPException(status_code=503, detail="model not loaded")
            recs = recommend_from_artifact(artifact, title=title, k=k)
        return RecommendResponse(query=title, k=k, recommendations=recs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"unexpected error: {e}")
