# AnimeOps – End-to-End MLOps Anime Recommender

AnimeOps is an end-to-end **MLOps project** that trains, evaluates, and serves a **content-based anime recommender system** using:

- **Kubeflow Pipelines** (running on k3s)
- **GitOps via GitHub Actions**
- **Docker** for training and serving images
- **Amazon S3** for data and model artifacts
- **FastAPI** for real-time inference
- **Kubernetes NodePort** for external access

This repository demonstrates a realistic production-style ML workflow, not a demo notebook.

---

## High-Level Architecture

```
CSV data (S3)
   ↓
Kubeflow Pipeline
   ├─ Train (TF-IDF recommender)
   ├─ Evaluate (quality gate)
   ↓
Model artifact (S3)
   ↓
FastAPI Serving App (Kubernetes)
   ↓
/recommend API (NodePort)
```

---

## Repository Structure

```
.
├── README.md
├── components
│   ├── training
│   │   ├── train.py
│   │   └── evaluate.py
│   └── serving
│       ├── app.py
│       └── Dockerfile
├── k8s
│   └── manifest.yaml
├── requirements.txt
├── scripts
│   ├── deploy.py
│   └── trigger_run.py
└── src
    ├── __init__.py
    └── pipeline.py
```

---

## How It Works

### Training & Evaluation
- Data is read from S3
- A TF-IDF model is trained
- Evaluation computes mean top-10 similarity
- Pipeline fails if quality threshold is not met

### CI/CD
- GitHub Actions builds Docker images
- Pipeline is compiled and uploaded to Kubeflow
- Images are referenced by commit SHA

### Serving
- FastAPI app runs in Kubernetes
- Model is downloaded from S3 on startup
- Exposed via NodePort service

---

## Using the Serving API

### Base URL

```
http://<NODE_IP>:30082
```

Example:

```
http://13.36.213.82:30082
```

### Health Check

```bash
curl http://<NODE_IP>:30082/healthz
```

- **200 OK** means the API is up and the model loaded successfully.
- **503** means the service is running but the model failed to load (check pod logs).

---

## Recommendation Endpoint

### Endpoint

- **Method:** `GET`
- **Path:** `/recommend`

### Query Parameters

#### `title` (required)
Anime title to base recommendations on.

- **Type:** string
- **Required:** yes
- **Example:** `Naruto`

Matching behavior:
- tries an **exact match** (case-insensitive)
- if not found, tries a **substring/contains match**
- if still not found, returns **404**

Example:

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto"
```

If the title is unknown:

```bash
curl "http://<NODE_IP>:30082/recommend?title=ThisDoesNotExist"
```

Returns (HTTP 404):

```json
{"detail":"Unknown title: ThisDoesNotExist"}
```

---

#### `k` (optional)
Number of recommendations to return.

- **Type:** integer
- **Default:** 10
- **Min:** 1
- **Max:** 50

Examples:

Return 5 results:

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto&k=5"
```

Return 20 results:

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto&k=20"
```

If you request too many (e.g. `k=500`), FastAPI will reject it with a validation error.

---

### Full Example

```bash
curl "http://<NODE_IP>:30082/recommend?title=Naruto&k=5"
```

Example response:

```json
{
  "query": "Naruto",
  "k": 5,
  "recommendations": [
    {
      "title": "Naruto Shippuden",
      "similarity": 0.42,
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

Field notes:
- `similarity` is the **model similarity score** (cosine similarity of TF-IDF vectors).
- `score`, `rank`, `popularity`, etc. come from the anime metadata dataset (if present).

---

### Useful Curl Tips

URL-encode titles with spaces/special characters:

```bash
curl -G "http://<NODE_IP>:30082/recommend" --data-urlencode "title=Princess Mononoke" --data-urlencode "k=5"
```

Pretty-print JSON (if you have `jq`):

```bash
curl -s "http://<NODE_IP>:30082/recommend?title=Naruto&k=5" | jq .
```

---

## Kubernetes Deployment

- Namespace: `animeops-serving`
- Service type: NodePort
- Port: `30082`
- IAM access via EC2 Instance Role

---

## Configuration

| Variable | Description |
|--------|------------|
| MODEL_S3_BUCKET | S3 bucket |
| MODEL_S3_KEY | Model path |
| MODEL_LOCAL_PATH | Local cache path for the model inside the container |
| DEFAULT_K | Default `k` used if not provided |
| MAX_K | Maximum allowed `k` |
| AWS_REGION / AWS_DEFAULT_REGION | AWS region used by boto3 |

---

## Troubleshooting (Serving)

### `{"detail":"model not loaded"}` (HTTP 503)
- Confirm the model exists in S3:
  ```bash
  aws s3 ls s3://mlops-anime-data/models/anime_recommender/model.joblib
  ```
- Check pod logs:
  ```bash
  kubectl -n animeops-serving logs deploy/animeops-serving
  ```
- Ensure your EC2 instance role has at least:
  - `s3:GetObject` on `arn:aws:s3:::mlops-anime-data/models/*`
  - `s3:ListBucket` on `arn:aws:s3:::mlops-anime-data`

---

## License
MIT
