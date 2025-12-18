import os
import time

import kfp
from kfp import Client

PIPELINE_PACKAGE = "anime_pipeline.yaml"
PIPELINE_NAME = "AnimeOps Pipeline"
EXPERIMENT_NAME = "AnimeOps"


def _pipeline_display_name(p):
    return getattr(p, "display_name", None) or getattr(p, "name", None)


def get_pipeline_id_by_name(client: Client, name: str):
    token = None
    while True:
        resp = client.list_pipelines(page_size=100, page_token=token)
        for p in (resp.pipelines or []):
            if _pipeline_display_name(p) == name:
                return getattr(p, "pipeline_id", None) or getattr(p, "id", None)
        token = getattr(resp, "next_page_token", None)
        if not token:
            return None


def get_or_create_experiment(client: Client, name: str):
    try:
        return client.get_experiment(experiment_name=name)
    except Exception:
        return client.create_experiment(name=name)


def main():
    host = os.getenv("KUBEFLOW_HOST")
    if not host:
        raise RuntimeError("KUBEFLOW_HOST is not set")

    client = Client(host=host)

    from src.pipeline import pipeline as pipeline_func

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=PIPELINE_PACKAGE,
    )

    pipeline_id = get_pipeline_id_by_name(client, PIPELINE_NAME)
    if pipeline_id:
        print(f"Pipeline exists: {PIPELINE_NAME} ({pipeline_id}). Uploading new version...")
        try:
            client.upload_pipeline_version(
                pipeline_package_path=PIPELINE_PACKAGE,
                pipeline_version_name=f"ver-{int(time.time())}",
                pipeline_id=pipeline_id,
            )
        except Exception:
            client.upload_pipeline(
                pipeline_package_path=PIPELINE_PACKAGE,
                pipeline_name=PIPELINE_NAME,
            )
    else:
        print(f"Pipeline not found. Creating: {PIPELINE_NAME}")
        client.upload_pipeline(
            pipeline_package_path=PIPELINE_PACKAGE,
            pipeline_name=PIPELINE_NAME,
        )

    if os.getenv("CREATE_RUN", "0") == "1":
        exp = get_or_create_experiment(client, EXPERIMENT_NAME)
        exp_id = getattr(exp, "experiment_id", None) or getattr(exp, "id", None)
        if not exp_id:
            raise RuntimeError(f"Could not determine experiment id from: {exp}")

        full_sha = os.getenv("GITHUB_SHA", "")
        sha7 = (full_sha or "manual")[:7]
        run_name = f"animeops-{sha7}-{int(time.time())}"

        arguments = {
            "bucket_name": os.getenv("BUCKET_NAME", "mlops-anime-data"),
            "data_key": os.getenv("DATA_KEY", "cleaned_anime_data.csv"),
            "runs_prefix": os.getenv("RUNS_PREFIX", "models/anime_recommender/runs"),
            "run_id": run_name,
            "git_sha": full_sha,

            "max_features": int(os.getenv("MAX_FEATURES", "50000")),
            "ngram_max": int(os.getenv("NGRAM_MAX", "2")),
            "min_df": int(os.getenv("MIN_DF", "2")),

            "eval_k": int(os.getenv("EVAL_K", "10")),
            "min_avg_genre_jaccard": float(os.getenv("MIN_AVG_GENRE_JACCARD", "0.40")),

            "promotion_metric_name": os.getenv("PROMOTION_METRIC_NAME", "avg_genre_jaccard_at_10"),
            "promotion_margin": float(os.getenv("PROMOTION_MARGIN", "0.00")),

            "approved_model_key": os.getenv(
                "APPROVED_MODEL_KEY",
                "models/anime_recommender/approved/model.joblib",
            ),
            "approved_meta_key": os.getenv(
                "APPROVED_META_KEY",
                "models/anime_recommender/approved/metadata.json",
            ),
        }

        print("Creating run with arguments:", arguments)

        run = client.create_run_from_pipeline_package(
            pipeline_file=PIPELINE_PACKAGE,
            run_name=run_name,
            experiment_id=exp_id,
            arguments=arguments,
        )
        run_id = getattr(run, "run_id", None) or getattr(run, "id", None)
        print(f"Created run: {run_id} ({run_name})")


if __name__ == "__main__":
    main()
