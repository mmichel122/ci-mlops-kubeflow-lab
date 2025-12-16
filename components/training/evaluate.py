import argparse
import joblib
import boto3
import numpy as np


def _s3_client():
    """Return a boto3 client for S3."""
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, dst_path: str) -> str:
    """Download a file from S3 to a local path."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    _s3_client().download_file(bucket, key, dst_path)
    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", required=True)
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--min_mean_similarity", type=float, required=True)
    args = parser.parse_args()

    # Download model from S3
    download_from_s3(args.bucket_name, args.model_key, "/tmp/model.joblib")

    # Load the model
    model = joblib.load("/tmp/model.joblib")
    tfidf = model["tfidf"]
    documents = model.get("documents", [])

    # Compute cosine similarity
    vectors = tfidf.transform(documents)
    sims = (vectors @ vectors.T).toarray()
    np.fill_diagonal(sims, 0)

    top10 = np.sort(sims, axis=1)[:, -10:]
    mean_sim = top10.mean()

    print(f"Mean top-10 similarity: {mean_sim}")

    # Check if model quality meets the minimum threshold
    if mean_sim < args.min_mean_similarity:
        raise RuntimeError(f"Model quality too low: {mean_sim} < {args.min_mean_similarity}")


if __name__ == "__main__":
    main()
