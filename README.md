# AnimeOps (Kubeflow + K3s)

Repo structure:
- `src/`: Kubeflow pipeline definition (KFP v2)
- `components/training/`: training container (downloads CSV from S3, trains TF‑IDF recommender, uploads model to S3)
- `components/serving/`: FastAPI inference service (loads model from S3, returns recommendations)
- `scripts/`: helper scripts for compiling + triggering runs

S3 bucket expected: `mlops-anime-data`
