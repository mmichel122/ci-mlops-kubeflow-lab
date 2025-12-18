import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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


def s3_copy_object(s3, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )


def _get_metric(d: Dict[str, Any], name: str) -> Optional[float]:
    v = d.get(name)
    if isinstance(v, (int, float)):
        return float(v)
    for path in (("eval", name), ("metrics", name), ("challenger", "metrics", name)):
        cur: Any = d
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bucket_name", required=True)

    ap.add_argument("--challenger_model_key", required=True)
    ap.add_argument("--eval_metrics_path", required=True)

    ap.add_argument("--approved_model_key", required=True)
    ap.add_argument("--approved_meta_key", required=True)

    ap.add_argument("--metric_name", default="avg_genre_jaccard_at_10")
    ap.add_argument("--margin", type=float, default=0.0)

    ap.add_argument("--run_id", default=os.getenv("RUN_ID", "unknown"))
    ap.add_argument("--git_sha", default=os.getenv("GITHUB_SHA", "manual"))
    ap.add_argument("--aws_region", default=os.getenv("AWS_REGION", ""))

    args = ap.parse_args()

    s3 = s3_client(args.aws_region or None)

    with open(args.eval_metrics_path, "r", encoding="utf-8") as f:
        challenger_metrics = json.load(f)

    challenger_val = _get_metric(challenger_metrics, args.metric_name)
    if challenger_val is None:
        raise RuntimeError(f"Missing metric '{args.metric_name}' in {args.eval_metrics_path}")

    approved_meta = s3_read_json(s3, args.bucket_name, args.approved_meta_key) or {}
    champion_val = _get_metric(approved_meta, args.metric_name)

    decision: Dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": args.metric_name,
        "margin": float(args.margin),
        "challenger": {
            "run_id": args.run_id,
            "git_sha": args.git_sha,
            "model_key": args.challenger_model_key,
            "metrics": challenger_metrics,
            "value": float(challenger_val),
        },
        "champion": {
            "model_key": args.approved_model_key,
            "value": float(champion_val) if champion_val is not None else None,
        },
    }

    if champion_val is None:
        should_promote = True
        reason = "no_existing_champion"
    else:
        required = champion_val * (1.0 + float(args.margin))
        should_promote = challenger_val >= required
        reason = "meets_margin" if should_promote else "below_margin"

    decision["promotion"] = {"promoted": bool(should_promote), "reason": reason}

    if should_promote:
        s3_copy_object(
            s3,
            args.bucket_name,
            src_key=args.challenger_model_key,
            dst_key=args.approved_model_key,
        )
        decision["promotion"]["promoted_at"] = datetime.now(timezone.utc).isoformat()

    new_meta = {
        **approved_meta,
        "metric_name": args.metric_name,
        "last_decision": decision,
    }
    if should_promote:
        new_meta["eval"] = challenger_metrics
        new_meta["champion"] = {
            "run_id": args.run_id,
            "git_sha": args.git_sha,
            "model_key": args.approved_model_key,
            "source_model_key": args.challenger_model_key,
            "promoted_at": decision["promotion"]["promoted_at"],
            "value": float(challenger_val),
        }

    s3_write_json(s3, args.bucket_name, args.approved_meta_key, new_meta)

    if should_promote:
        print(f"Promoted model to s3://{args.bucket_name}/{args.approved_model_key}")
    else:
        print("Did not promote model (kept current champion).")
    print(f"Metadata: s3://{args.bucket_name}/{args.approved_meta_key}")


if __name__ == "__main__":
    main()
