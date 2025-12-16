import argparse
import os
import joblib
import pandas as pd
import boto3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict, Any
from scipy import sparse

def _s3_client():
    """Return a boto3 client for S3."""
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, dst_path: str) -> str:
    """Download a file from S3 to a local path."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    _s3_client().download_file(bucket, key, dst_path)
    return dst_path


def upload_to_s3(src_path: str, bucket: str, key: str) -> str:
    """Upload a local file to S3."""
    _s3_client().upload_file(src_path, bucket, key)
    return f"s3://{bucket}/{key}"


def normalize_title(row: pd.Series) -> str:
    """Normalize the title by checking various fields (English, Japanese, Synonyms)."""
    for col in ("English", "Japanese", "Synonyms"):
        val = row.get(col, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(row.name)


def build_corpus(df: pd.DataFrame) -> List[str]:
    """Combine different fields into one text blob for each anime."""
    def _get(col: str) -> pd.Series:
        return df.get(col, pd.Series([""] * len(df)))

    parts = [
        _get("Description"),
        _get("Genres"),
        _get("Studios"),
        _get("Producers"),
        _get("Type"),
        _get("Source"),
        _get("Demographic"),
        _get("Rating"),
    ]
    corpus = (parts[0] + " | " + parts[1] + " | " + parts[2] + " | " + parts[3]
              + " | " + parts[4] + " | " + parts[5] + " | " + parts[6] + " | " + parts[7]).tolist()
    return corpus


@dataclass
class RecommenderModel:
    """A dataclass for the recommender model."""
    titles: List[str]
    tfidf: sparse.csr_matrix
    vectorizer: TfidfVectorizer
    meta: pd.DataFrame

    def recommend(self, title: str, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k recommendations for a given anime title."""
        title_norm = title.strip().lower()
        titles_lower = [t.lower() for t in self.titles]
        idx = titles_lower.index(title_norm) if title_norm in titles_lower else None

        if idx is None:
            for i, t in enumerate(titles_lower):
                if title_norm in t:
                    idx = i
                    break

        if idx is None:
            raise ValueError(f"Unknown title: {title}")

        sims = cosine_similarity(self.tfidf[idx], self.tfidf).ravel()
        sims[idx] = -1.0  # Exclude itself
        top_idx = np.argsort(-sims)[:k]

        recommendations = []
        for j in top_idx:
            rec = {"title": self.titles[j], "score": float(sims[j])}
            for col in ("Score", "Popularity", "Rank", "Type", "Episodes", "Studios", "Genres"):
                if col in self.meta.columns:
                    val = self.meta.iloc[j][col]
                    if pd.notna(val):
                        rec[col.lower()] = val if not isinstance(val, (np.generic,)) else val.item()
            recommendations.append(rec)

        return recommendations


def train(df: pd.DataFrame, max_features: int = 50000, ngram_max: int = 2) -> RecommenderModel:
    """Train a content-based recommender model."""
    df["__title__"] = df.apply(normalize_title, axis=1)
    df = df.drop_duplicates(subset="__title__", keep="first").reset_index(drop=True)

    corpus = build_corpus(df)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1, ngram_max), min_df=2)
    tfidf = vectorizer.fit_transform(corpus)
    meta_cols = [c for c in df.columns if c not in ("Description",)]
    meta = df[meta_cols].copy()

    return RecommenderModel(titles=df["__title__"].tolist(), tfidf=tfidf, vectorizer=vectorizer, meta=meta)


def compute_basic_metrics(df: pd.DataFrame, model: RecommenderModel) -> Dict[str, Any]:
    """Compute basic metrics for the model."""
    metrics = {
        "num_items": len(model.titles),
        "tfidf_nonzero": model.tfidf.nnz,
        "avg_title_len": np.mean([len(t) for t in model.titles]) if model.titles else 0.0
    }
    if "Score" in df.columns:
        metrics["avg_score"] = float(pd.to_numeric(df["Score"], errors="coerce").dropna().mean())
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
    args = ap.parse_args()

    download_from_s3(args.bucket_name, args.data_key, args.local_data_path)
    df = pd.read_csv(args.local_data_path)

    model = train(df, max_features=args.max_features, ngram_max=args.ngram_max)

    os.makedirs(os.path.dirname(args.local_model_path), exist_ok=True)
    joblib.dump(model, args.local_model_path)

    model_uri = upload_to_s3(args.local_model_path, args.bucket_name, args.model_key)

    metrics = compute_basic_metrics(df, model)
    metrics["model_uri"] = model_uri
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    import json
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(model_uri)


if __name__ == "__main__":
    main()
