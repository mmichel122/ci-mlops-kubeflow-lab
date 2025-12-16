\
from kfp import dsl
from kfp import compiler


@dsl.container_component
def train_recommender(
    bucket_name: str,
    data_key: str,
    model_key: str,
    max_features: int,
    ngram_max: int,
    metrics_path: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image="REPLACE_WITH_TRAINING_IMAGE",  # e.g. 548894310305.dkr.ecr.eu-west-3.amazonaws.com/anime-ops-training:latest
        command=["python", "-m", "components.training.train"],
        args=[
            "--bucket_name", bucket_name,
            "--data_key", data_key,
            "--model_key", model_key,
            "--max_features", max_features,
            "--ngram_max", ngram_max,
            "--metrics_path", metrics_path,
        ],
    )


@dsl.pipeline(
    name="animeops-training-pipeline",
    description="Train a simple content-based anime recommender and store the model in S3.",
)
def anime_pipeline(
    bucket_name: str = "mlops-anime-data",
    data_key: str = "cleaned_anime_data.csv",
    model_key: str = "models/anime_recommender/model.joblib",
    max_features: int = 50000,
    ngram_max: int = 2,
):
    train_task = train_recommender(
        bucket_name=bucket_name,
        data_key=data_key,
        model_key=model_key,
        max_features=max_features,
        ngram_max=ngram_max,
    )

    train_task.set_cpu_request("500m").set_cpu_limit("2")
    train_task.set_memory_request("1Gi").set_memory_limit("4Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(anime_pipeline, package_path="anime_pipeline.yaml")
