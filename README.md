# AnimeOps – End-to-End MLOps Anime Recommender

AnimeOps is an end-to-end **MLOps project** that trains, evaluates, and serves a **content-based anime recommender** using:

- **Kubeflow Pipelines (KFP v2)** running on **k3s** (on an AWS EC2 instance)
- **GitOps** via **GitHub Actions** with a **self-hosted runner** on the k3s server
- **Docker** images for training and serving
- **Amazon S3** for datasets, models, and UI artifacts (**`mlops-anime-data`**)
- **FastAPI** for real-time inference
- **Kubernetes NodePort** for external access

This repo is structured like a production workflow: reproducible pipeline runs, artifact versioning in S3, and a serving stack that can be promoted via an “approved model” pointer.

---

## High-Level Architecture

```text
CSV data (S3: mlops-anime-data)
   ↓
Kubeflow Pipeline
   ├─ Train (TF-IDF vectorization + cosine similarity index)
   ├─ Evaluate (quality gate + diversity checks)
   ├─ Promote (copy/run-pointer → approved model key)
   ↓
Model artifact (S3)
   ↓
FastAPI Serving App (Kubernetes)
   ├─ initContainer syncs UI from S3 (optional)
   └─ API downloads approved model from S3 + caches it
   ↓
/recommend API (NodePort)
```

---

## Repository Structure

```text
.
├── README.md
├── components
│   ├── training
│   │   ├── train.py
│   │   └── evaluate.py
│   └── serving
│       ├── app.py
│       ├── Dockerfile
│       └── ui/                  # (optional) static UI assets
├── k8s
│   └── manifest.yaml            # Deployment + Service (+ initContainer for UI sync)
├── requirements.txt
├── scripts
│   ├── deploy.py
│   └── trigger_run.py
└── src
    ├── __init__.py
    └── pipeline.py
```

---

## Data

Dataset columns used for recommendations and enrichment typically include:

- **Description** (main text feature)
- **Genres**, **Type**, **Episodes**, **Studios/Producers** (optional enrichers)
- **Score**, **Rank**, **Popularity**, **Members** (returned as metadata)

### Cleaning notes

Before training, the dataset should be cleaned to:
- Drop rows missing **Genres** (if you use it in evaluation/diversity logic)
- Coerce **Episodes** to numeric and drop invalid rows
- Remove duplicated/echoed genre tokens (e.g., “ActionAction” → “Action”)

---

## Model

### What the “model” is

AnimeOps is a **content-based retrieval model**:

1. Text feature engineering using **TF‑IDF**
2. Similarity search using **cosine similarity** between TF‑IDF vectors

The saved artifact (Joblib) typically contains:
- `vectorizer` (the fitted `TfidfVectorizer`)
- `tfidf_matrix` (sparse matrix of all anime vectors)
- `titles` / metadata table needed for lookup + response enrichment
- run metadata (e.g., `run_id`, `git_sha`, S3 ETag/version info)

### Core TF‑IDF hyperparameters (current defaults)

- `max_features = 50000`  
  Caps vocabulary size to the top-N terms by corpus statistics (controls memory and speed).
- `ngram_max = 2`  
  Uses unigrams + bigrams, capturing phrases like “spirit world”.
- `min_df = 2`  
  Drops terms that appear in fewer than 2 documents (removes noisy/one-off tokens).

---

## Kubeflow Pipeline

### Train step
- Reads CSV from S3
- Builds TF‑IDF vectors for each title
- Writes `model.joblib` back to S3 under a **run-scoped key**
- Emits training metrics as a **KFP Metrics artifact**

Example run output pattern:
```text
s3://mlops-anime-data/models/anime_recommender/runs/<run_id>/model.joblib
```

### Evaluate step (quality gate)
Evaluation is designed to stop bad models from being promoted.

Typical checks include:
- `k`: top‑K list length used during evaluation
- `min_avg_genre_jaccard`: minimum average genre Jaccard similarity for recommendations
- **Coverage/Diversity**: % of unique titles that appear in anyone’s top‑K  
  (prevents recommending the same small handful of anime for many queries)

If evaluation fails, the pipeline run fails and promotion does not happen.

### Promote step (approved model pointer)
When evaluation passes, the pipeline promotes the run artifact by copying (or updating a pointer) to:

```text
s3://mlops-anime-data/models/anime_recommender/approved/model.joblib
```

Serving always loads from this **approved** key so production can be updated safely.

---

## CI/CD (GitOps)

### What GitHub Actions does
- Builds/pushes Docker images (training + serving) tagged by commit SHA
- Compiles and uploads the KFP pipeline
- Triggers a run when relevant code changes

### Change-based workflow optimization
The workflow should **not** retrain on every push. Use path filtering to only run:
- training build + pipeline run when `components/training/**` or pipeline code changes
- serving build/deploy when `components/serving/**` or `k8s/**` changes
- UI sync when `components/serving/ui/**` changes

This avoids unnecessary training runs when only the serving/UI is modified.

---

## Serving

### Serving components
- **FastAPI** service that:
  - downloads the approved model from S3 (and caches it locally in the container)
  - supports case-insensitive exact match, then substring match for titles
  - returns enriched metadata fields (score/rank/popularity/etc.) if present

- (Optional) **static UI**
  - Built in-repo under `components/serving/ui/`
  - Synced to S3 via:
    ```bash
    aws s3 sync ./ui s3://mlops-anime-data/ui/animeops/ --delete
    ```
    Notes:
    - Uploads changed/new files
    - Removes files from S3 that no longer exist locally (`--delete`)
    - If `index.html` in the repo is newer/changed, it will be replaced in S3 on next sync

  - Pulled into the pod via an **initContainer** (e.g., `sync-ui-from-s3`) into a shared volume.

---

## Using the Serving API

### Base URL

```text
http://<NODE_IP>:30082
```

Example:

```text
http://13.36.213.82:30082
```

### Health Check

```bash
curl http://<NODE_IP>:30082/healthz
```

- **200 OK**: API up and model loaded
- **503**: service running but model failed to load (check pod logs)

---

## Recommendation Endpoint

### Endpoint

- **Method:** `GET`
- **Path:** `/recommend`

### Query parameters

#### `title` (required)
Anime title to base recommendations on.

Matching behavior:
- exact match (case-insensitive)
- fallback: substring/contains match
- else: **404**

Examples:

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto"
curl "http://<NODE_IP>:30082/recommend?title=ThisDoesNotExist"
```

If the title has spaces/special characters, URL-encode it:

```bash
curl -G "http://<NODE_IP>:30082/recommend" \
  --data-urlencode "title=Princess Mononoke" \
  --data-urlencode "k=5"
```

#### `k` (optional)
Number of recommendations to return.

- Default: `10`
- Min: `1`
- Max: `50` (validated by the API)

Examples:

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto&k=5"
curl "http://<NODE_IP>:30082/recommend?title=Naruto&k=20"
```

### Example response

```json
{
  "query": "Naruto",
  "k": 5,
  "recommendations": [
    {
      "title": "Naruto Shippuden",
      "similarity": 0.4630,
      "score": 8.27,
      "popularity": 16,
      "rank": 294,
      "type": "TV",
      "episodes": 500,
      "studios": "Pierrot",
      "genres": "Action, Adventure, Fantasy"
    }
  ]
}
```

---

## Kubernetes Deployment

- Namespace: `animeops-serving`
- Service type: NodePort
- NodePort: `30082`
- AWS auth: **EC2 Instance Role** (recommended) so pods can read from S3 without static credentials

---

## Configuration

| Variable | Description |
|---|---|
| `MODEL_S3_BUCKET` | S3 bucket name (default: `mlops-anime-data`) |
| `MODEL_S3_KEY` | S3 key for the approved model (default: `models/anime_recommender/approved/model.joblib`) |
| `MODEL_LOCAL_PATH` | Local cache path inside container (default: `/tmp/model/model.joblib`) |
| `DEFAULT_K` | Default `k` when not provided |
| `MAX_K` | Max allowed `k` |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | AWS region for boto3 |

---

## Troubleshooting

### Model not loaded (HTTP 503)
1. Confirm the approved model exists in S3:
   ```bash
   aws s3 ls s3://mlops-anime-data/models/anime_recommender/approved/
   ```
2. Check pod logs:
   ```bash
   kubectl -n animeops-serving logs deploy/animeops-serving
   ```
3. Ensure the EC2 instance role allows:
   - `s3:GetObject` on `arn:aws:s3:::mlops-anime-data/models/*`
   - `s3:ListBucket` on `arn:aws:s3:::mlops-anime-data`

### UI not updating
- Confirm the UI files are in S3:
  ```bash
  aws s3 ls s3://mlops-anime-data/ui/animeops/
  ```
- If using an initContainer, restart the deployment to re-run the init sync:
  ```bash
  kubectl -n animeops-serving rollout restart deploy/animeops-serving
  ```

---

## Notes on production hardening (next steps)
- Add a lightweight **model registry** JSON in S3 (run metadata + metrics) and promote by updating a pointer.
- Add **canary** deployment for serving (two deployments with different approved keys).
- Add more metrics: latency, request volume, error rates, and recommendation diversity per time window.
