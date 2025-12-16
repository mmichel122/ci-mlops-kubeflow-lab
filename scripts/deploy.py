\
"""
Compile pipeline and (optionally) upload to Kubeflow Pipelines.
This script is intentionally minimal for GitOps flows: you can run it in CI to regenerate `anime_pipeline.yaml`.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    # compile by running src/pipeline.py
    subprocess.check_call([sys.executable, str(ROOT / "src" / "pipeline.py")])
    print("Wrote", ROOT / "anime_pipeline.yaml")

if __name__ == "__main__":
    main()
