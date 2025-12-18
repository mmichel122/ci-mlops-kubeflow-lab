import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def s3_client():
    return boto3.client("s3")


def head_s3_object(bucket: str, key: str) -> Dict[str, Any]:
    s3 = s3_client()
    obj = s3.head_object(Bucket=bucket, Key=key)
    etag = (obj.get("ETag") or "").strip('"')
    return {
        "bucket": bucket,
        "key": key,
        "etag": etag,
        "version_id": obj.get("VersionId"),
        "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
        "content_length": int(obj.get("ContentLength") or 0),
    }


def download_from_s3(bucket: str, key: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    s3_client().download_file(bucket, key, dst_path)
    return dst_path


def upload_to_s3(src_path: str, bucket: str, key: str) -> str:
    s3_client().upload_file(src_path, bucket, key)
    return f"s3://{bucket}/{key}"


def normalize_title(row: pd.Series) -> str:
    for col in ("English Title", "Japanese Title", "English", "Japanese", "Synonyms"):
        if col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(row.name)


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].fillna("").astype(str)


def build_corpus(df: pd.DataFrame) -> List[str]:
    parts = [
        _safe_col(df, "Description"),
        _safe_col(df, "Genres"),
        _safe_col(df, "Studios"),
        _safe_col(df, "Producers"),
        _safe_col(df, "Type"),
        _safe_col(df, "Source"),
        _safe_col(df, "Demographic"),
        _safe_col(df, "Rating"),
    ]
    corpus = (
        parts[0]
        + " | " + parts[1]
        + " | " + parts[2]
        + " | " + parts[3]
        + " | " + parts[4]
        + " | " + parts[5]
        + " | " + parts[6]
        + " | " + parts[7]
    ).tolist()
    return [c if isinstance(c, str) else "" for c in corpus]


def compute_basic_metrics(df: pd.DataFrame, X) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "num_items": int(len(df)),
        "tfidf_nonzero": int(getattr(X, "nnz", 0)),
    }
    if "Score" in df.columns:
        s = pd.to_numeric(df["Score"], errors="coerce").dropna()
        if not s.empty:
            metrics["avg_score"] = float(s.mean())
    if hasattr(X, "shape"):
        metrics["tfidf_shape"] = [int(X.shape[0]), int(X.shape[1])]
    return metrics


def _hash_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bucket_name", required=True)
    ap.add_argument("--data_key", required=True)

    # Storage layout
    ap.add_argument("--runs_prefix", default="models/anime_recommender/runs")
    ap.add_argument(
        "--model_key",
        default="",
        help="If set, writes the model to this exact key. Otherwise uses runs_prefix/<run_id>/model.joblib",
    )

    # Provenance (optional)
    ap.add_argument("--run_id", default=os.getenv("RUN_ID", ""))
    ap.add_argument("--git_sha", default=os.getenv("GITHUB_SHA", ""))

    # Artifacts on the container FS
    ap.add_argument("--metrics_path", default="/outputs/train_metrics.json")
    ap.add_argument("--local_data_path", default="/tmp/data/cleaned_anime_data.csv")
    ap.add_argument("--local_model_path", default="/tmp/model/model.joblib")

    # Hyperparameters
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)

    # KFP primitive outputs (optional)
    ap.add_argument("--model_key_out", default="")
    ap.add_argument("--run_id_out", default="")
    ap.add_argument("--data_etag_out", default="")

    args = ap.parse_args()

    data_meta = head_s3_object(args.bucket_name, args.data_key)

    params = {
        "max_features": int(args.max_features),
        "ngram_max": int(args.ngram_max),
        "min_df": int(args.min_df),
        "stop_words": "english",
    }

    # Run id: user-supplied, else deterministic from params+data+time
    if args.run_id and args.run_id.strip():
        run_id = args.run_id.strip()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + _hash_id(
            {"params": params, "data": {"key": args.data_key, "etag": data_meta.get("etag")}, "git": args.git_sha}
        )

    runs_prefix = args.runs_prefix.strip("/")

    if args.model_key and args.model_key.strip():
        model_key = args.model_key.strip("/")
    else:
        model_key = f"{runs_prefix}/{run_id}/model.joblib"

    # Download + load
    download_from_s3(args.bucket_name, args.data_key, args.local_data_path)
    df = pd.read_csv(args.local_data_path)

    # Normalize titles + dedupe
    df = df.copy()
    df["__title__"] = df.apply(normalize_title, axis=1)
    df = df.drop_duplicates(subset="__title__", keep="first").reset_index(drop=True)

    corpus = build_corpus(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
    )
    X = vectorizer.fit_transform(corpus)

    created_at = datetime.now(timezone.utc).isoformat()

    artifact: Dict[str, Any] = {
        "titles": df["__title__"].tolist(),
        "vectorizer": vectorizer,
        "tfidf_matrix": X,
        "meta": df[[c for c in df.columns if c != "Description"]].to_dict(orient="list"),
        "schema_version": 2,
        "params": params,
        "provenance": {
            "run_id": run_id,
            "git_sha": args.git_sha or None,
            "created_at": created_at,
            "data": data_meta,
        },
    }

    os.makedirs(os.path.dirname(args.local_model_path), exist_ok=True)
    joblib.dump(artifact, args.local_model_path)

    model_uri = upload_to_s3(args.local_model_path, args.bucket_name, model_key)

    # Write train metrics locally (KFP artifact path)
    metrics = compute_basic_metrics(df, X)
    metrics.update(
        {
            "model_uri": model_uri,
            "model_key": model_key,
            "run_id": run_id,
            "created_at": created_at,
            "git_sha": args.git_sha or None,
            "data": data_meta,
            **params,
        }
    )

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Also upload a params+train_metrics snapshot next to the model
    tmp_dir = "/tmp/runmeta"
    os.makedirs(tmp_dir, exist_ok=True)
    params_path = os.path.join(tmp_dir, "params.json")
    train_metrics_path = os.path.join(tmp_dir, "train_metrics.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "params": params, "data": data_meta, "git_sha": args.git_sha or None},
            f,
            indent=2,
        )
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    upload_to_s3(params_path, args.bucket_name, f"{runs_prefix}/{run_id}/params.json")
    upload_to_s3(train_metrics_path, args.bucket_name, f"{runs_prefix}/{run_id}/train_metrics.json")

    # KFP primitive outputs (optional)
    if args.model_key_out:
        os.makedirs(os.path.dirname(args.model_key_out), exist_ok=True)
        with open(args.model_key_out, "w", encoding="utf-8") as f:
            f.write(model_key)

    if args.run_id_out:
        os.makedirs(os.path.dirname(args.run_id_out), exist_ok=True)
        with open(args.run_id_out, "w", encoding="utf-8") as f:
            f.write(run_id)

    if args.data_etag_out:
        os.makedirs(os.path.dirname(args.data_etag_out), exist_ok=True)
        with open(args.data_etag_out, "w", encoding="utf-8") as f:
            f.write(str(data_meta.get("etag") or ""))

    print(model_uri)


if __name__ == "__main__":
    main()
