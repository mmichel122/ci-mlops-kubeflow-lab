import argparse
import json
import os
from typing import Any, Dict, List

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def s3_client():
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    s3_client().download_file(bucket, key, dst_path)
    return dst_path


def upload_to_s3(src_path: str, bucket: str, key: str) -> str:
    s3_client().upload_file(src_path, bucket, key)
    return f"s3://{bucket}/{key}"


def normalize_title(row: pd.Series) -> str:
    # Adjust to your actual columns if needed
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
    # Use only columns you *might* have; missing cols become ""
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
    # join row-wise
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

    # final safety: ensure no NaNs remain
    corpus = [c if isinstance(c, str) else "" for c in corpus]
    return corpus


def compute_basic_metrics(df: pd.DataFrame, X) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "num_items": int(len(df)),
        "tfidf_nonzero": int(getattr(X, "nnz", 0)),
    }
    if "Score" in df.columns:
        s = pd.to_numeric(df["Score"], errors="coerce").dropna()
        if not s.empty:
            metrics["avg_score"] = float(s.mean())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_name", required=True)
    ap.add_argument("--data_key", required=True)
    ap.add_argument("--model_key", default="models/anime_recommender/model.joblib")
    ap.add_argument("--metrics_path", default="/outputs/metrics.json")

    ap.add_argument("--local_data_path", default="/tmp/data/cleaned_anime_data.csv")
    ap.add_argument("--local_model_path", default="/tmp/model/model.joblib")

    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--min_df", type=int, default=2)
    args = ap.parse_args()

    download_from_s3(args.bucket_name, args.data_key, args.local_data_path)
    df = pd.read_csv(args.local_data_path)

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

    # Save a robust artifact (dict), not a custom class
    artifact: Dict[str, Any] = {
        "titles": df["__title__"].tolist(),
        "vectorizer": vectorizer,
        "tfidf_matrix": X,  # sparse matrix is fine in joblib
        # Keep only light metadata to avoid huge pickles
        "meta": df[[c for c in df.columns if c != "Description"]].to_dict(orient="list"),
        "schema_version": 1,
    }

    os.makedirs(os.path.dirname(args.local_model_path), exist_ok=True)
    joblib.dump(artifact, args.local_model_path)

    model_uri = upload_to_s3(args.local_model_path, args.bucket_name, args.model_key)

    metrics = compute_basic_metrics(df, X)
    metrics.update(
        {
            "model_uri": model_uri,
            "max_features": int(args.max_features),
            "ngram_max": int(args.ngram_max),
            "min_df": int(args.min_df),
        }
    )

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(model_uri)


if __name__ == "__main__":
    main()
