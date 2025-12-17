import os
import time
import kfp
from kfp import Client

PIPELINE_PACKAGE = "anime_pipeline.yaml"
PIPELINE_NAME = "AnimeOps Pipeline"
EXPERIMENT_NAME = "AnimeOps"


def get_pipeline_id_by_name(client: Client, name: str):
    token = None
    while True:
        resp = client.list_pipelines(page_size=100, page_token=token)
        for p in (resp.pipelines or []):
            if p.name == name:
                return p.id
        token = getattr(resp, "next_page_token", None)
        if not token:
            return None


def get_or_create_experiment(client: Client, name: str):
    # KFP client often supports get_experiment(experiment_name=...)
    try:
        exp = client.get_experiment(experiment_name=name)
        return exp
    except Exception:
        return client.create_experiment(name=name)


def main():
    host = os.getenv("KUBEFLOW_HOST")
    if not host:
        raise RuntimeError("KUBEFLOW_HOST is not set")

    from src.pipeline import pipeline as pipeline_func

    # Compile
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=PIPELINE_PACKAGE,
    )
    print(f"Wrote {os.path.abspath(PIPELINE_PACKAGE)}")

    client = Client(host=host)
    print(f"Connected to KFP at {host}")

    pipeline_id = get_pipeline_id_by_name(client, PIPELINE_NAME)

    # Upload pipeline or version
    version_name = f"{PIPELINE_NAME}-{os.getenv('GITHUB_SHA','manual')[:7]}-{int(time.time())}"

    if pipeline_id:
        print(f"Pipeline exists: {PIPELINE_NAME} (id={pipeline_id}). Uploading version: {version_name}")
        client.upload_pipeline_version(
            pipeline_package_path=PIPELINE_PACKAGE,
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id,
        )
    else:
        print(f"Pipeline not found. Creating: {PIPELINE_NAME}")
        client.upload_pipeline(
            pipeline_package_path=PIPELINE_PACKAGE,
            pipeline_name=PIPELINE_NAME,
        )
        pipeline_id = get_pipeline_id_by_name(client, PIPELINE_NAME)

        # First version upload is optional; KFP usually creates a default version.
        print("Created pipeline, continuing...")

    # Create run (GitOps-style)
    if os.getenv("CREATE_RUN", "0") == "1":
        exp = get_or_create_experiment(client, EXPERIMENT_NAME)

        sha = os.getenv("GITHUB_SHA", "manual")[:7]
        run_name = f"animeops-{sha}-{int(time.time())}"

        # Make challenger key unique per run (immutable)
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
            experiment_id=exp.experiment_id,
            arguments=arguments,
        )
        print(f"Created run: {run.run_id} ({run_name})")


if __name__ == "__main__":
    main()
