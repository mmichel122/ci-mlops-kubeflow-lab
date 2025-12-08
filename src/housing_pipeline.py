from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics
import logging

# Configure Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- COMPONENT 1: INGESTION (Dynamic) ---
@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'boto3'])
def ingest_data(
    output_csv: Output[Dataset],
    bucket_name: str,  # <--- Dynamic Input
    object_key: str    # <--- Dynamic Input
):
    import boto3
    import pandas as pd
    import logging
    from io import BytesIO
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Connecting to S3 (Using Instance Profile)...")
    
    try:
        s3 = boto3.client('s3') 
        logger.info(f"Downloading s3://{bucket_name}/{object_key}...")
        
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        df = pd.read_csv(obj['Body'])
        
        df.to_csv(output_csv.path, index=False)
        logger.info(f"Data ingested successfully: {len(df)} rows.")
        
    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")
        raise e

# --- COMPONENT 2: DRIFT DETECTION ---
@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'scipy', 'boto3'])
def detect_drift(
    new_data: Input[Dataset],
    reference_bucket: str,
    reference_key: str,
    drift_threshold: float = 0.05
) -> str:
    import pandas as pd
    import boto3
    import logging
    from scipy.stats import ks_2samp
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("[Drift] Starting Drift Detection...")
    
    # 1. Load New Data
    df_new = pd.read_csv(new_data.path)
    
    # 2. Load Reference Data
    logger.info(f"[Drift] Fetching Reference Data from s3://{reference_bucket}/{reference_key}...")
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=reference_bucket, Key=reference_key)
        df_ref = pd.read_csv(obj['Body'])
    except Exception as e:
        logger.warning(f"[Drift] Reference data not found ({e}). Defaulting to DRIFT.")
        return "true"
    
    # 3. Calculate Drift
    logger.info("[Drift] Running Kolmogorov-Smirnov Test on 'price' column...")
    
    if 'price' not in df_new.columns or 'price' not in df_ref.columns:
        logger.warning("[Drift] 'price' column missing. Defaulting to DRIFT.")
        return "true"
        
    statistic, p_value = ks_2samp(df_new['price'], df_ref['price'])
    
    logger.info(f"   KS Statistic: {statistic:.4f}")
    logger.info(f"   P-Value: {p_value:.4f}")
    
    if p_value < drift_threshold:
        logger.info(f"[Drift] DETECTED! (P-Value {p_value:.4f} < {drift_threshold})")
        return "true"
    else:
        logger.info(f"[Drift] Data is stable. (P-Value {p_value:.4f} >= {drift_threshold})")
        return "false"

# --- COMPONENT 3: TRAINING ---
@dsl.component(base_image='python:3.9', packages_to_install=['pandas', 'scikit-learn', 'joblib'])
def train_model(dataset: Input[Dataset], model_artifact: Output[Model]):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading data...")
    df = pd.read_csv(dataset.path)
    
    df = df.rename(columns={"sqft_living": "sqft", "bedrooms": "bedrooms", "grade": "grade", "price": "price"})
    if 'is_high_value' not in df.columns:
        df['is_high_value'] = (df['price'] > 500000).astype(int)
    
    X = df[['sqft', 'bedrooms', 'grade']]
    y = df['is_high_value']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump(model, model_artifact.path)
    logger.info(f"Model saved.")

# --- COMPONENT 4: EVALUATION ---
@dsl.component(
    base_image='python:3.9', 
    packages_to_install=['pandas', 'scikit-learn', 'joblib', 'mlflow==2.14.0', 'matplotlib', 'boto3']
)
def evaluate_model(
    dataset: Input[Dataset], 
    model_artifact: Input[Model], 
    metrics: Output[Metrics],
    mlflow_tracking_uri: str,
    experiment_name: str
):
    import pandas as pd
    import joblib
    import mlflow
    import logging
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Connecting to MLflow at {mlflow_tracking_uri}...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="KFP-Auto-Run"):
        df = pd.read_csv(dataset.path)
        model = joblib.load(model_artifact.path)
        
        df = df.rename(columns={"sqft_living": "sqft", "price": "price"})
        if 'is_high_value' not in df.columns:
            df['is_high_value'] = (df['price'] > 500000).astype(int)
        
        X = df[['sqft', 'bedrooms', 'grade']]
        y = df['is_high_value']
        
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs)
        
        logger.info(f"Metrics: Accuracy={acc:.4f}, AUC={auc:.4f}")
        
        metrics.log_metric("accuracy", acc)
        metrics.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        
        logger.info("Uploading model to MLflow...")
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "Housing_KFP_Model")

# --- PIPELINE WIRING ---
@dsl.pipeline(
    name='housing-event-driven-pipeline',
    description='Pipeline triggered by S3 Events with Drift Detection'
)
def housing_pipeline(
    bucket_name: str = "mlops-housing-data-drift",
    object_key: str = "data/housing.csv",
    mlflow_url: str = "http://mlflow-server.kubeflow.svc.cluster.local:5000",
    experiment_name: str = "Kubeflow_Housing_Runs"
):
    # 1. Ingest (Using arguments from Trigger)
    task_data = ingest_data(
        bucket_name=bucket_name,
        object_key=object_key
    )
    
    # 2. Drift Check
    # We check against a baseline file (e.g. the original training data)
    task_drift = detect_drift(
        new_data=task_data.outputs['output_csv'],
        reference_bucket=bucket_name,
        reference_key="data/housing.csv"
    )
    
    # 3. Conditional Training
    # Only run if Drift == "true"
    with dsl.Condition(task_drift.output == "true", name="drift-detected"):
        
        task_train = train_model(dataset=task_data.outputs['output_csv'])
        
        task_eval = evaluate_model(
            dataset=task_data.outputs['output_csv'],
            model_artifact=task_train.outputs['model_artifact'],
            mlflow_tracking_uri=mlflow_url,
            experiment_name=experiment_name
        )

if __name__ == "__main__":
    # 1. Compile
    package_path = 'housing_pipeline_gitops.yaml'
    compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path=package_path
    )
    logger.info(f"✅ Compiled to {package_path}")

    # 2. Deploy (The missing part!)
    import kfp
    import sys
    
    # We use localhost because the Runner is ON the same machine as the cluster
    client = kfp.Client(host='http://localhost:8080')
    
    logger.info("🚀 Auto-Submitting to Kubeflow...")
    
    try:
        # We use create_run_from_pipeline_package to upload & run in one shot
        run = client.create_run_from_pipeline_package(
            pipeline_file=package_path,
            arguments={
                "bucket_name": "mlops-housing-data-drift",
                "object_key": "data/housing.csv"
            },
            run_name="Git-Triggered-Run",
            enable_caching=True
        )
        logger.info(f"✅ Pipeline Successfully Submitted! Run ID: {run.run_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to submit pipeline: {e}")
        sys.exit(1) # Fail the CI/CD job