import argparse
import json
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", required=True)
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--min_mean_similarity", type=float, required=True)
    parser.add_argument("--metrics_path", default="/outputs/eval_metrics.json")
    args = parser.parse_args()

    local_model_path = "/tmp/model/model.joblib"
    download_from_s3(args.bucket_name, args.model_key, local_model_path)

    artifact = joblib.load(local_model_path)
    X = artifact["tfidf_matrix"]

    # Sparse cosine similarity; returns sparse if dense_output=False
    S = cosine_similarity(X, X, dense_output=False)

    topk_vals = []
    for i in range(S.shape[0]):
        row = S.getrow(i).tocsr()

        # remove self
        if row.shape[1] > i:
            row = row.copy()
            row[0, i] = 0.0
            row.eliminate_zeros()

        vals = row.data
        if vals.size == 0:
            continue
        k = min(10, vals.size)
        topk = np.partition(vals, -k)[-k:]
        topk_vals.append(topk)

    if not topk_vals:
        raise RuntimeError("Evaluation failed: no similarities computed (empty top-k).")

    mean_sim = float(np.mean(np.concatenate(topk_vals)))
    print("Mean top-10 similarity:", mean_sim)

    eval_metrics = {"mean_top10_similarity": mean_sim, "num_items": int(S.shape[0])}
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    if mean_sim < args.min_mean_similarity:
        raise RuntimeError(
            f"Model quality too low: {mean_sim} < {args.min_mean_similarity}"
        )


if __name__ == "__main__":
    main()
