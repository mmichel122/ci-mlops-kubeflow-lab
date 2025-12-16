\
import os
from kfp import Client

PIPELINE_YAML = os.getenv("PIPELINE_YAML", "anime_pipeline.yaml")
KFP_HOST = os.getenv("KFP_HOST")  # e.g. http://<kubeflow-pipelines-endpoint>
EXPERIMENT_NAME = os.getenv("KFP_EXPERIMENT", "AnimeOps")
RUN_NAME = os.getenv("KFP_RUN_NAME", "animeops-train-run")


def main():
    if not KFP_HOST:
        raise SystemExit("Set KFP_HOST (e.g. http://<kubeflow-pipelines-endpoint>)")

    client = Client(host=KFP_HOST)

    exp = client.create_experiment(name=EXPERIMENT_NAME)
    run = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_YAML,
        experiment_id=exp.experiment_id,
        run_name=RUN_NAME,
        arguments={
            "bucket_name": os.getenv("BUCKET_NAME", "mlops-anime-data"),
            "data_key": os.getenv("DATA_KEY", "cleaned_anime_data.csv"),
            "model_key": os.getenv("MODEL_KEY", "models/anime_recommender/model.joblib"),
            "max_features": int(os.getenv("MAX_FEATURES", "50000")),
            "ngram_max": int(os.getenv("NGRAM_MAX", "2")),
        },
    )
    print("Run created:", run.run_id)


if __name__ == "__main__":
    main()
