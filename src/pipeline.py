import os
from typing import Annotated
from kfp import dsl


def _training_image() -> str:
    img = os.getenv("TRAINING_IMAGE")
    if not img:
        raise ValueError("TRAINING_IMAGE env var is not set (CI must provide it).")
    return img


@dsl.container_component
def train_recommender(
    bucket_name: str,
    data_key: str,
    model_key: str,
    max_features: int,
    ngram_max: int,
    metrics_path: Annotated[str, dsl.OutputPath()],
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
    metrics_path: Annotated[str, dsl.OutputPath()],
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


@dsl.container_component
def promote_model(
    bucket_name: str,
    challenger_model_key: str,
    eval_metrics_path: Annotated[str, dsl.InputPath()],
    approved_model_key: str,
    approved_meta_key: str,
    margin: float,
):
    return dsl.ContainerSpec(
        image=_training_image(),
        command=["python", "promote.py"],
        args=[
            "--bucket_name", bucket_name,
            "--challenger_model_key", challenger_model_key,
            "--eval_metrics_path", eval_metrics_path,
            "--approved_model_key", approved_model_key,
            "--approved_meta_key", approved_meta_key,
            "--margin", str(margin),
            "--metric_name", "mean_top10_similarity",
        ],
    )


@dsl.pipeline(
    name="animeops-training-pipeline",
    description="Train + evaluate + promote (champion/challenger) a TF-IDF anime recommender.",
)
def pipeline(
    bucket_name: str = "mlops-anime-data",
    data_key: str = "cleaned_anime_data.csv",

    # challenger output (unique per run)
    model_key: str = "models/anime_recommender/runs/manual/model.joblib",

    # stable champion pointers
    approved_model_key: str = "models/anime_recommender/approved/model.joblib",
    approved_meta_key: str = "models/anime_recommender/approved/metadata.json",

    max_features: int = 50000,
    ngram_max: int = 2,

    min_mean_similarity: float = 0.15,
    promotion_margin: float = 0.005,
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

    promote_task = promote_model(
        bucket_name=bucket_name,
        challenger_model_key=model_key,
        eval_metrics_path=eval_task.outputs["metrics_path"],
        approved_model_key=approved_model_key,
        approved_meta_key=approved_meta_key,
        margin=promotion_margin,
    ).after(eval_task)
    promote_task.set_cpu_request("100m").set_cpu_limit("500m")
    promote_task.set_memory_request("128Mi").set_memory_limit("512Mi")
