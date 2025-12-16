import os
import kfp
from kfp import Client

PIPELINE_PACKAGE = "anime_pipeline.yaml"
PIPELINE_NAME = "AnimeOps Pipeline"
EXPERIMENT_NAME = "AnimeOps"

def main():
    host = os.getenv("KUBEFLOW_HOST")
    if not host:
        raise RuntimeError("KUBEFLOW_HOST is not set")

    # Import pipeline function from src.pipeline
    from src import pipeline as pipeline_module

    # Accept either `pipeline` or `animeops_pipeline`
    if hasattr(pipeline_module, "pipeline"):
        pipeline_func = pipeline_module.pipeline
    elif hasattr(pipeline_module, "animeops_pipeline"):
        pipeline_func = pipeline_module.animeops_pipeline
    else:
        raise AttributeError(
            "src/pipeline.py must define a pipeline function named `pipeline` "
            "or `animeops_pipeline`."
        )

    # Compile
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=PIPELINE_PACKAGE,
    )
    print(f"Wrote {os.path.abspath(PIPELINE_PACKAGE)}")

    # Connect
    client = Client(host=host)
    print(f"Connected to KFP at {host}")

    # Upload (creates pipeline if missing)
    client.upload_pipeline(
        pipeline_package_path=PIPELINE_PACKAGE,
        pipeline_name=PIPELINE_NAME,
    )
    print(f"Uploaded pipeline: {PIPELINE_NAME}")

    # Optional run
    if os.getenv("CREATE_RUN", "0") == "1":
        exp = client.create_experiment(name=EXPERIMENT_NAME)
        run = client.create_run_from_pipeline_package(
            pipeline_file=PIPELINE_PACKAGE,
            run_name=f"animeops-{os.getenv('GITHUB_SHA','manual')[:7]}",
            experiment_id=exp.experiment_id,
            arguments={},
        )
        print(f"Created run: {run.run_id}")

if __name__ == "__main__":
    main()
