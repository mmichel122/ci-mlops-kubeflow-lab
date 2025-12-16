import os
import sys
import importlib.util
import kfp
from kfp import Client

PIPELINE_PACKAGE = "anime_pipeline.yaml"
PIPELINE_NAME = "AnimeOps Pipeline"
EXPERIMENT_NAME = "AnimeOps"

def load_pipeline_func():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pipeline_path = os.path.join(repo_root, "src", "pipeline.py")

    spec = importlib.util.spec_from_file_location("animeops_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline module from {pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["animeops_pipeline"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "pipeline"):
        raise AttributeError("src/pipeline.py must define a function named `pipeline`")

    return module.pipeline

def main():
    host = os.getenv("KUBEFLOW_HOST")
    if not host:
        raise RuntimeError("KUBEFLOW_HOST is not set")

    pipeline_func = load_pipeline_func()

    # compile
    kfp.compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PIPELINE_PACKAGE)
    print(f"Wrote {os.path.abspath(PIPELINE_PACKAGE)}")

    # connect
    client = Client(host=host)
    print(f"Connected to KFP at {host}")

    # upload
    client.upload_pipeline(
        pipeline_package_path=PIPELINE_PACKAGE,
        pipeline_name=PIPELINE_NAME,
    )
    print(f"Uploaded pipeline: {PIPELINE_NAME}")

    # optional run
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
