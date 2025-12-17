import argparse
import json
import os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


def s3_client(region: str | None = None):
    if region:
        return boto3.client("s3", region_name=region)
    return boto3.client("s3")


def s3_read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response["Error"].get("Code", "")
        if code in ("NoSuchKey", "404"):
            return None
        raise


def s3_write_json(s3, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket_name", required=True)

    # Challenger inputs (produced by this run)
    ap.add_argument("--challenger_model_key", required=True)
    ap.add_argument("--eval_metrics_path", required=True)

    # Champion pointers (stable)
    ap.add_argument("--approved_model_key", required=True)
    ap.add_argument("--approved_meta_key", required=True)

    # Metric comparison
    ap.add_argument("--metric_name", default="mean_top10_similarity")
    ap.add_argument("--margin", type=float, default=0.005)

    # Optional metadata
    ap.add_argument("--run_id", default=os.getenv("RUN_ID", "unknown"))
    ap.add_argument("--git_sha", default=os.getenv("GITHUB_SHA", "manual"))
    ap.add_argument("--aws_region", default=os.getenv("AWS_REGION", ""))

    args = ap.parse_args()

    s3 = s3_client(args.aws_region or None)

    with open(args.eval_metrics_path, "r", encoding="utf-8") as f:
        challenger_metrics = json.load(f)

    if args.metric_name not in challenger_metrics:
        raise RuntimeError(
            f"Missing metric '{args.metric_name}' in {args.eval_metrics_path}. "
            f"Found keys: {list(challenger_metrics.keys())}"
        )

    challenger_score = float(challenger_metrics[args.metric_name])

    approved_meta = s3_read_json(s3, args.bucket_name, args.approved_meta_key)
    champion_score = None
    if approved_meta and args.metric_name in approved_meta:
        champion_score = float(approved_meta[args.metric_name])

    # Decide
    should_promote = False
    if champion_score is None:
        should_promote = True  # first model becomes champion
    else:
        should_promote = challenger_score > (champion_score + args.margin)

    decision = {
        "promoted": should_promote,
        "metric_name": args.metric_name,
        "challenger_score": challenger_score,
        "champion_score": champion_score,
        "margin": args.margin,
        "challenger_model_key": args.challenger_model_key,
        "approved_model_key": args.approved_model_key,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "git_sha": args.git_sha,
    }

    print("PROMOTION_DECISION:", json.dumps(decision, indent=2))

    if not should_promote:
        return

    # Copy challenger model to approved pointer
    copy_source = {"Bucket": args.bucket_name, "Key": args.challenger_model_key}
    s3.copy_object(
        Bucket=args.bucket_name,
        Key=args.approved_model_key,
        CopySource=copy_source,
    )

    # Write/update approved metadata
    new_meta = {
        **(approved_meta or {}),
        **decision,
    }
    s3_write_json(s3, args.bucket_name, args.approved_meta_key, new_meta)

    print(f"Promoted model to s3://{args.bucket_name}/{args.approved_model_key}")
    print(f"Updated metadata at s3://{args.bucket_name}/{args.approved_meta_key}")


if __name__ == "__main__":
    main()
