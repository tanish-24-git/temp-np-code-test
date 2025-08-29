from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # File Upload Configuration
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = [".csv", ".xlsx", ".json"]
    upload_directory: str = "uploads"
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ]
    
    # Processing Configuration
    max_chunk_size: int = 10000
    default_test_size: float = 0.2
    random_state: int = 42
    
    # Cloud Storage Configuration
    cloud_storage_provider: str = "local"
    aws_s3_bucket: str = ""
    aws_region: str = "us-east-1"
    
    # ML Configuration
    default_hyperparameters: Dict = {
        "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "logistic_regression": {"max_iter": [500, 1000, 2000]},
        "kmeans": {"n_clusters": [2, 3, 4, 5]}
    }
    max_training_time: int = 3600
    
    # Celery Configuration
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()