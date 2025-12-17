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

    from src.pipeline import pipeline as pipeline_func

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=PIPELINE_PACKAGE,
    )
    print(f"Wrote {os.path.abspath(PIPELINE_PACKAGE)}")

    client = Client(host=host)
    print(f"Connected to KFP at {host}")

    pipeline_id = get_pipeline_id_by_name(client, PIPELINE_NAME)
    version_name = f"{PIPELINE_NAME}-{os.getenv('GITHUB_SHA','manual')[:7]}-{int(time.time())}"

    if pipeline_id:
        print(f"Pipeline exists: {PIPELINE_NAME} (id={pipeline_id}). Uploading version: {version_name}")

        if hasattr(client, "upload_pipeline_version"):
            client.upload_pipeline_version(
                pipeline_package_path=PIPELINE_PACKAGE,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id,
            )
        else:
            # Fallback: many KFP v2 installs treat re-upload as a new version
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
        pipeline_id = get_pipeline_id_by_name(client, PIPELINE_NAME)
        print("Created pipeline, continuing...")

    if os.getenv("CREATE_RUN", "0") == "1":
        exp = get_or_create_experiment(client, EXPERIMENT_NAME)
        exp_id = getattr(exp, "experiment_id", None) or getattr(exp, "id", None)
        if not exp_id:
            raise RuntimeError(f"Could not determine experiment id from: {exp}")

        sha = os.getenv("GITHUB_SHA", "manual")[:7]
        run_name = f"animeops-{sha}-{int(time.time())}"
        model_key = f"models/anime_recommender/runs/{run_name}/model.joblib"

        arguments = {
            "bucket_name": os.getenv("BUCKET_NAME", "mlops-anime-data"),
            "data_key": os.getenv("DATA_KEY", "cleaned_anime_data.csv"),
            "model_key": model_key,
            "approved_model_key": "models/anime_recommender/approved/model.joblib",
            "approved_meta_key": "models/anime_recommender/approved/metadata.json",
            "min_mean_similarity": float(os.getenv("MIN_MEAN_SIM", "0.15")),
            "promotion_margin": float(os.getenv("PROMOTION_MARGIN", "0.005")),
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
