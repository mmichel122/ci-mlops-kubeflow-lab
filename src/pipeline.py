import kfp
from kfp import dsl

def anime_training_op(alpha: float, bucket_name: str, data_key: str):
    # This defines the Container that runs on K3s
    return dsl.ContainerOp(
        name='Train Anime Recommender',
        # REPLACE THIS with your actual AWS ECR URI
        image='123456789012.dkr.ecr.us-east-1.amazonaws.com/anime-ops:latest', 
        arguments=[
            '--alpha', alpha,
            '--bucket_name', bucket_name,
            '--data_key', data_key
        ],
        # We assume the code inside handles the S3 upload/MLflow logging
    ).set_image_pull_policy('Always') # Important for development!

@dsl.pipeline(
    name='Anime Recommender Training',
    description='Downloads data from S3, trains hybrid model, and logs to MLflow.'
)
def anime_pipeline(
    alpha: float = 0.8,
    bucket_name: str = 'mlops-anime-data',
    data_key: str = 'cleaned_anime_data.csv'
):
    # Create the task
    train_task = anime_training_op(alpha, bucket_name, data_key)
    
    # K3s Optimization: Set resource limits to avoid crashing your node
    train_task.set_memory_request('500Mi')
    train_task.set_cpu_request('500m')
    
    # If using AWS credentials on K3s, we need to inject the secret
    # train_task.apply(kfp.aws.use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

if __name__ == '__main__':
    # Compiles the pipeline to YAML when run directly
    kfp.compiler.Compiler().compile(anime_pipeline, 'anime_pipeline.yaml')