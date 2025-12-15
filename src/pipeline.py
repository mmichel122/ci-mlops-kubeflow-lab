from kfp import dsl
from kfp import compiler

@dsl.container_component
def anime_training_op(alpha: float, bucket_name: str, data_key: str):
    return dsl.ContainerSpec(
        # REPLACE with your actual ECR URI
        image='548894310305.dkr.ecr.eu-west-3.amazonaws.com/anime-ops:latest',
        args=[
            '--alpha', alpha,
            '--bucket_name', bucket_name,
            '--data_key', data_key
        ]
    )

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
    train_task = anime_training_op(alpha=alpha, bucket_name=bucket_name, data_key=data_key)
    
    # KFP v2 Resource limits
    train_task.set_memory_request('500Mi')
    train_task.set_cpu_request('500m')

if __name__ == '__main__':
    compiler.Compiler().compile(anime_pipeline, 'anime_pipeline.yaml')