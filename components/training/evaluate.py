import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Set

import boto3
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def s3_client():
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    s3_client().download_file(bucket, key, dst_path)
    return dst_path


def upload_to_s3(src_path: str, bucket: str, key: str) -> str:
    s3_client().upload_file(src_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _parse_genres(raw: str) -> Set[str]:
    if not isinstance(raw, str):
        return set()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return set(parts)


def _avg_genre_jaccard_at_k(genres: List[str], topk_idx: np.ndarray) -> float:
    scores = []
    for i in range(topk_idx.shape[0]):
        g_i = _parse_genres(genres[i])
        if not g_i:
            continue
        for j in topk_idx[i]:
            g_j = _parse_genres(genres[int(j)])
            if not g_j:
                continue
            inter = len(g_i & g_j)
            union = len(g_i | g_j)
            if union > 0:
                scores.append(inter / union)
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_name", required=True)
    ap.add_argument("--model_key", required=True)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min_avg_genre_jaccard", type=float, default=0.0)

    # If provided, also uploads the eval metrics next to the run folder
    ap.add_argument("--runs_prefix", default="models/anime_recommender/runs")
    ap.add_argument("--run_id", default=os.getenv("RUN_ID", ""))

    ap.add_argument("--local_model_path", default="/tmp/model/model.joblib")
    ap.add_argument("--metrics_path", default="/outputs/eval_metrics.json")
    args = ap.parse_args()

    download_from_s3(args.bucket_name, args.model_key, args.local_model_path)
    artifact = joblib.load(args.local_model_path)

    X = artifact["tfidf_matrix"]
    titles = artifact.get("titles", [])
    meta = artifact.get("meta", {}) or {}
    genres = meta.get("Genres") or [""] * len(titles)

    created_at = datetime.now(timezone.utc).isoformat()

    # Similarity matrix
    S = cosine_similarity(X)
    n = S.shape[0]

    # Effective k (cannot recommend more than n-1 items)
    k_eff = min(int(args.k), max(n - 1, 1))

    # Exclude self from retrieval
    np.fill_diagonal(S, -np.inf)

    # Get top-k_eff indices (unsorted) per row: shape (n, k_eff)
    topk = np.argpartition(-S, kth=k_eff - 1, axis=1)[:, :k_eff]

    # Sort those indices by similarity descending using take_along_axis
    topk_sims = np.take_along_axis(S, topk, axis=1)      # (n, k_eff)
    order = np.argsort(-topk_sims, axis=1)               # (n, k_eff)
    topk = np.take_along_axis(topk, order, axis=1)       # (n, k_eff)
    topk_sims = np.take_along_axis(S, topk, axis=1)      # (n, k_eff)

    mean_topk_similarity = float(topk_sims.mean()) if topk_sims.size else 0.0
    avg_genre_jaccard = _avg_genre_jaccard_at_k(genres, topk)

    eval_metrics = {
        "created_at": created_at,
        "model_key": args.model_key,
        "run_id": args.run_id or artifact.get("provenance", {}).get("run_id"),
        "num_items": int(n),
        "k": int(k_eff),
        f"avg_genre_jaccard_at_{k_eff}": float(avg_genre_jaccard),
        f"mean_top{k_eff}_similarity": float(mean_topk_similarity),
    }

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    # Optional upload
    run_id = eval_metrics.get("run_id") or ""
    if run_id:
        runs_prefix = args.runs_prefix.strip("/")
        tmp = "/tmp/runmeta/eval_metrics.json"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        upload_to_s3(tmp, args.bucket_name, f"{runs_prefix}/{run_id}/eval_metrics.json")

    metric_name = f"avg_genre_jaccard_at_{k_eff}"
    if eval_metrics[metric_name] < args.min_avg_genre_jaccard:
        raise RuntimeError(
            f"Model quality too low: {metric_name}={eval_metrics[metric_name]:.4f} < {args.min_avg_genre_jaccard:.4f}"
        )


if __name__ == "__main__":
    main()
