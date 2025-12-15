import kfp
import os
import sys

# Configuration
PIPELINE_FILE = 'anime_pipeline.yaml'
PIPELINE_NAME = 'Anime Recommender Training'
KUBEFLOW_HOST = 'http://localhost:8080/pipeline' # Adjust if using port-forwarding

def deploy():
    print(f"1. Compiling Pipeline...")
    # Compile src/pipeline.py -> anime_pipeline.yaml
    exit_code = os.system(f"dsl-compile --py src/pipeline.py --output {PIPELINE_FILE}")
    if exit_code != 0:
        print("Compilation Failed!")
        sys.exit(1)

    print(f"2. Connecting to Kubeflow at {KUBEFLOW_HOST}...")
    client = kfp.Client(host=KUBEFLOW_HOST)

    # Check if the pipeline exists
    existing_pipelines = client.list_pipelines(filter=dict(name=PIPELINE_NAME))
    
    if existing_pipelines.pipelines:
        pipeline_id = existing_pipelines.pipelines[0].id
        print(f"3. Found existing pipeline ID: {pipeline_id}. Uploading new version...")
        
        # Upload a new version to the existing pipeline
        # Use a version name based on git commit or timestamp usually
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