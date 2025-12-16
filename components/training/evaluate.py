import argparse
import joblib
import numpy as np
import boto3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", required=True)
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--min_mean_similarity", type=float, required=True)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    s3.download_file(args.bucket_name, args.model_key, "/tmp/model.joblib")

    model = joblib.load("/tmp/model.joblib")
    tfidf = model["tfidf"]
    vectors = tfidf.transform(model["documents"])

    sims = (vectors @ vectors.T).toarray()
    np.fill_diagonal(sims, 0)

    top10 = np.sort(sims, axis=1)[:, -10:]
    mean_sim = float(top10.mean())

    print("Mean top-10 similarity:", mean_sim)

    if mean_sim < args.min_mean_similarity:
        raise RuntimeError(
            f"Model quality too low: {mean_sim} < {args.min_mean_similarity}"
        )

if __name__ == "__main__":
    main()
