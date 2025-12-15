import kfp
import sys
import os

# Import your pipeline function directly
# We need to add the project root to path so we can import src.pipeline
sys.path.append(os.getcwd())
from src.pipeline import anime_pipeline

# Configuration
PIPELINE_FILE = 'anime_pipeline.yaml'
PIPELINE_NAME = 'Anime Recommender Training'
KUBEFLOW_HOST = 'http://10.43.248.45/pipeline'

def deploy():
    print(f"1. Compiling Pipeline (KFP v2)...")
    
    # Use the Python SDK compiler, NOT os.system
    from kfp.compiler import Compiler
    Compiler().compile(pipeline_func=anime_pipeline, package_path=PIPELINE_FILE)
    
    print(f"2. Connecting to Kubeflow at {KUBEFLOW_HOST}...")
    client = kfp.Client(host=KUBEFLOW_HOST)

    # Check if pipeline exists
    # Note: KFP v2 client might return different structures, this is a robust check
    existing_pipelines = client.list_pipelines(filter=dict(name=PIPELINE_NAME))
    
    if existing_pipelines.pipelines:
        pipeline_id = existing_pipelines.pipelines[0].id
        print(f"3. Found existing pipeline ID: {pipeline_id}. Uploading new version...")
        
        import time
        version_name = f"v-{int(time.time())}"
        
        client.upload_pipeline_version(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id
        )
        print(f"Success! Version {version_name} deployed.")
    else:
        print("3. Pipeline not found. Creating new pipeline...")
        client.upload_pipeline(
            pipeline_package_path=PIPELINE_FILE,
            pipeline_name=PIPELINE_NAME
        )
        print("Success! New pipeline created.")

if __name__ == "__main__":
    deploy()