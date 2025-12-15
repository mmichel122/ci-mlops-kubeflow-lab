import argparse
import pandas as pd
import pickle
import numpy as np
import os
import boto3
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Argument Parsing (Standard for Kubeflow Components) ---
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.8, help='Weight for plot vs genre')
parser.add_argument('--bucket_name', type=str, required=True, help='S3 Bucket Name')
parser.add_argument('--data_key', type=str, default='cleaned_anime_data.csv', help='S3 Key')
args = parser.parse_args()

def download_from_s3(bucket, key, local_path):
    print(f"Downloading s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)

def train(df, alpha):
    print(f"Training with Alpha={alpha}...")
    
    # 1. Feature Engineering: Description (TF-IDF)
    df['Description'] = df['Description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    desc_vectors = tfidf.fit_transform(df['Description'])
    
    # 2. Feature Engineering: Genres
    df['genre_list'] = df['Genres'].apply(
        lambda x: [g.strip() for g in str(x).split(',')] if pd.notnull(x) else []
    )
    mlb = MultiLabelBinarizer()
    genre_vectors = mlb.fit_transform(df['genre_list'])
    
    # 3. Similarity Calculation
    sim_plot = cosine_similarity(desc_vectors)
    sim_genre = cosine_similarity(genre_vectors)
    
    # Hybrid Weighting
    final_sim = (alpha * sim_plot) + ((1 - alpha) * sim_genre)
    
    return final_sim, mlb, tfidf

if __name__ == "__main__":
    # Enable MLflow Autologging (Captures system metrics)
    mlflow.autolog(disable=True)
    
    with mlflow.start_run():
        # Log Params
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("dataset", f"s3://{args.bucket_name}/{args.data_key}")
        
        # 1. Load Data
        local_data = "dataset.csv"
        try:
            download_from_s3(args.bucket_name, args.data_key, local_data)
        except Exception as e:
            print(f"S3 Download Failed: {e}")
            # Fallback for local testing if S3 fails
            if os.path.exists("../../data/cleaned_anime_data.csv"):
                print("Falling back to local data...")
                import shutil
                shutil.copy("../../data/cleaned_anime_data.csv", local_data)
            else:
                raise e

        df = pd.read_csv(local_data)
        
        # 2. Train
        sim_matrix, mlb, tfidf = train(df, args.alpha)
        
        # 3. Save Artifacts locally
        os.makedirs("model_artifacts", exist_ok=True)
        
        with open("model_artifacts/similarity_matrix.pkl", "wb") as f:
            pickle.dump(sim_matrix, f)
            
        # Save Metadata (Critical for mapping Index -> Anime Title)
        metadata = df[['Score', 'Rank', 'Popularity', 'Rating', 'English', 'Japanese', 'Genres']].reset_index(drop=True)
        metadata.to_pickle("model_artifacts/metadata.pkl")
        
        # 4. Log to MLflow
        print("Logging artifacts to MLflow...")
        mlflow.log_artifact("model_artifacts/similarity_matrix.pkl")
        mlflow.log_artifact("model_artifacts/metadata.pkl")
        
        # Log a custom metric: "Coverage" (how many anime have at least 1 similar item > 0.5 score)
        # This helps detect if our model is too strict
        coverage = np.mean(np.max(sim_matrix, axis=1) > 0.5)
        mlflow.log_metric("similarity_coverage", coverage)
        
        print("Training Run Complete.")