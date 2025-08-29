from typing import List, Dict
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
import json

class Settings(BaseSettings):
    api_host: str = Field("127.0.0.1", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")

    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")  # consistent with usage
    cloud_provider: str = Field("aws", env="CLOUD_PROVIDER")
    aws_bucket: str = Field(..., env="AWS_BUCKET")  # required

    aws_region: str = Field("us-east-1", env="AWS_REGION")
    upload_directory: str = Field("./uploads", env="UPLOAD_DIRECTORY")

    allowed_extensions: List[str] = Field(default_factory=lambda: [".csv", ".xlsx", ".json"], env="ALLOWED_EXTENSIONS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], env="CORS_ORIGINS")

    default_test_size: float = Field(0.2, env="DEFAULT_TEST_SIZE")
    random_state: int = Field(42, env="RANDOM_STATE")

    default_hyperparameters: Dict = Field(
        default_factory=lambda: {
            "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "logistic_regression": {"max_iter": [500, 1000, 2000]},
            "kmeans": {"n_clusters": [2, 3, 4, 5]},
        }
    )

    max_training_time: int = Field(3600, env="MAX_TRAINING_TIME")

    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model_name: str = Field("llama-3.3", env="GROQ_MODEL_NAME")

    @field_validator("allowed_extensions", mode="before")
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("cors_origins", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        validate_default = True  # Ensures defaults are validated

settings = Settings()
