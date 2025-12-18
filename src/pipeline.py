import os
from kfp import dsl
from kfp.dsl import Input, Output, Metrics, OutputPath


def _training_image() -> str:
    img = os.getenv("TRAINING_IMAGE")
    if not img:
        raise ValueError("TRAINING_IMAGE env var is not set (CI must provide it).")
    return img


@dsl.container_component
def train_recommender(
    bucket_name: str,
    data_key: str,
    runs_prefix: str,
    run_id: str,
    git_sha: str,
    max_features: int,
    ngram_max: int,
    min_df: int,
    train_metrics: Output[Metrics],
    model_key_out: OutputPath(str),
    run_id_out: OutputPath(str),
    data_etag_out: OutputPath(str),
):
    return dsl.ContainerSpec(
        image=_training_image(),
        command=["python", "train.py"],
        args=[
            "--bucket_name", bucket_name,
            "--data_key", data_key,
            "--runs_prefix", runs_prefix,
            "--run_id", run_id,
            "--git_sha", git_sha,
            "--max_features", str(max_features),
            "--ngram_max", str(ngram_max),
            "--min_df", str(min_df),
            "--metrics_path", train_metrics.path,
            "--model_key_out", model_key_out,
            "--run_id_out", run_id_out,
            "--data_etag_out", data_etag_out,
        ],
    )


@dsl.container_component
def evaluate_recommender(
    bucket_name: str,
    model_key: str,
    runs_prefix: str,
    run_id: str,
    k: int,
    min_avg_genre_jaccard: float,
    eval_metrics: Output[Metrics],
):
    return dsl.ContainerSpec(
        image=_training_image(),
        command=["python", "evaluate.py"],
        args=[
            "--bucket_name", bucket_name,
            "--model_key", model_key,
            "--runs_prefix", runs_prefix,
            "--run_id", run_id,
            "--k", str(k),
            "--min_avg_genre_jaccard", str(min_avg_genre_jaccard),
            "--metrics_path", eval_metrics.path,
        ],
    )


@dsl.container_component
def promote_model(
    bucket_name: str,
    challenger_model_key: str,
    eval_metrics: Input[Metrics],
    approved_model_key: str,
    approved_meta_key: str,
    metric_name: str,
    margin: float,
    run_id: str,
    git_sha: str,
):
    return dsl.ContainerSpec(
        image=_training_image(),
        command=["python", "promote.py"],
        args=[
            "--bucket_name", bucket_name,
            "--challenger_model_key", challenger_model_key,
            "--eval_metrics_path", eval_metrics.path,
            "--approved_model_key", approved_model_key,
            "--approved_meta_key", approved_meta_key,
            "--metric_name", metric_name,
            "--margin", str(margin),
            "--run_id", run_id,
            "--git_sha", git_sha,
        ],
    )


@dsl.pipeline(name="animeops-recommender")
def pipeline(
    bucket_name: str = "mlops-anime-data",
    data_key: str = "cleaned_anime_data.csv",

    runs_prefix: str = "models/anime_recommender/runs",
    run_id: str = "",
    git_sha: str = "",

    approved_model_key: str = "models/anime_recommender/approved/model.joblib",
    approved_meta_key: str = "models/anime_recommender/approved/metadata.json",

    max_features: int = 50000,
    ngram_max: int = 2,
    min_df: int = 2,

    eval_k: int = 10,
    min_avg_genre_jaccard: float = 0.40,
    promotion_metric_name: str = "avg_genre_jaccard_at_10",
    promotion_margin: float = 0.00,
):
    train_task = train_recommender(
        bucket_name=bucket_name,
        data_key=data_key,
        runs_prefix=runs_prefix,
        run_id=run_id,
        git_sha=git_sha,
        max_features=max_features,
        ngram_max=ngram_max,
        min_df=min_df,
    )
    train_task.set_cpu_request("500m").set_cpu_limit("2")
    train_task.set_memory_request("1Gi").set_memory_limit("4Gi")

    eval_task = evaluate_recommender(
        bucket_name=bucket_name,
        model_key=train_task.outputs["model_key_out"],
        runs_prefix=runs_prefix,
        run_id=train_task.outputs["run_id_out"],
        k=eval_k,
        min_avg_genre_jaccard=min_avg_genre_jaccard,
    ).after(train_task)
    eval_task.set_cpu_request("250m").set_cpu_limit("1")
    eval_task.set_memory_request("512Mi").set_memory_limit("2Gi")

    promote_task = promote_model(
        bucket_name=bucket_name,
        challenger_model_key=train_task.outputs["model_key_out"],
        eval_metrics=eval_task.outputs["eval_metrics"],
        approved_model_key=approved_model_key,
        approved_meta_key=approved_meta_key,
        metric_name=promotion_metric_name,
        margin=promotion_margin,
        run_id=train_task.outputs["run_id_out"],
        git_sha=git_sha,
    ).after(eval_task)
    promote_task.set_cpu_request("100m").set_cpu_limit("500m")
    promote_task.set_memory_request("128Mi").set_memory_limit("512Mi")
