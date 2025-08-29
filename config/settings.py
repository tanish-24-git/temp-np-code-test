from pydantic_settings import BaseSettings
from pydantic import validator
from typing import List, Dict
import json


class Settings(BaseSettings):
    api_host: str
    api_port: int
    debug: bool

    max_file_size_mb: int  # matches MAX_FILE_SIZE_MB
    cloud_storage_provider: str  # matches CLOUD_STORAGE_PROVIDER
    aws_s3_bucket: str  # matches AWS_S3_BUCKET
    
    # Other fields, renamed for consistent snake_case + full match with env vars:
    aws_region: str
    allowed_extensions: List[str]
    upload_directory: str

    cors_origins: List[str]

    default_test_size: float
    random_state: int

    default_hyperparameters: Dict = {
        "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "logistic_regression": {"max_iter": [500, 1000, 2000]},
        "kmeans": {"n_clusters": [2, 3, 4, 5]},
    }
    max_training_time: int = 3600

    celery_broker_url: str
    celery_result_backend: str

    @validator('allowed_extensions', pre=True)
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("allowed_extensions must be a valid JSON list string")
        return v

    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("cors_origins must be a valid JSON list string")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
