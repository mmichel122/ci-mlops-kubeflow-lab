import os
from kfp import dsl


def _training_image() -> str:
    img = os.getenv("TRAINING_IMAGE")
    if not img:
        raise ValueError(
            "TRAINING_IMAGE env var is not set. "
            "Set it in GitHub Actions before running scripts/deploy.py."
        )
    return img


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
        image=_training_image(),
        command=["python", "train.py"],
        args=[
            "--bucket_name", bucket_name,
            "--data_key", data_key,
            "--model_key", model_key,
            "--max_features", str(max_features),
            "--ngram_max", str(ngram_max),
            "--metrics_path", metrics_path,
        ],
    )


@dsl.container_component
def evaluate_recommender(
    bucket_name: str,
    model_key: str,
    min_mean_similarity: float,
    metrics_path: dsl.OutputPath(str),
):
    return dsl.ContainerSpec(
        image=_training_image(),
        command=["python", "evaluate.py"],
        args=[
            "--bucket_name", bucket_name,
            "--model_key", model_key,
            "--min_mean_similarity", str(min_mean_similarity),
            "--metrics_path", metrics_path,
        ],
    )


@dsl.pipeline(
    name="animeops-training-pipeline",
    description="Train + evaluate a simple content-based anime recommender and store the model in S3.",
)
def pipeline(
    bucket_name: str = "mlops-anime-data",
    data_key: str = "cleaned_anime_data.csv",
    model_key: str = "models/anime_recommender/model.joblib",
    max_features: int = 50000,
    ngram_max: int = 2,
    min_mean_similarity: float = 0.15,
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

    eval_task = evaluate_recommender(
        bucket_name=bucket_name,
        model_key=model_key,
        min_mean_similarity=min_mean_similarity,
    ).after(train_task)
    eval_task.set_cpu_request("250m").set_cpu_limit("1")
    eval_task.set_memory_request("512Mi").set_memory_limit("2Gi")
