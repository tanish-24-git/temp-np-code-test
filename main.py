import os
import json
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from models.responses import UploadResponse, PreprocessResponse, TrainResponse
from services.file_service import FileService
from services.preprocessing_service import PreprocessingService
from services.ml_service import MLService, AsyncMLService
from services.llm_service import LLMService  # New import
from utils.validators import file_validator, validate_preprocessing_params
from utils.exceptions import DatasetError, PreprocessingError, ModelTrainingError

from tasks import train_model_task  # Note: still present but can be adjusted to sync call if desired


app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for training ML models without code",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_service = FileService()
preprocessing_service = PreprocessingService()
ml_service = MLService()
async_ml_service = AsyncMLService()
llm_service = LLMService()  # Initialize LLM service

os.makedirs(settings.upload_directory, exist_ok=True)


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload datasets, analyze them, and get AI-driven recommendations for preprocessing and modeling.
    """
    results = {}
    try:
        for file in files:
            print(f"Processing file: {file.filename}")
            await file_validator.validate_file(file)
            file_path = await file_service.save_uploaded_file(file)

            # Perform dataset analysis
            analysis = file_service.analyze_dataset(file_path)

            # Load a sample of dataset (e.g., 20 rows) for enhanced LLM prompt
            import pandas as pd
            df_sample = pd.read_csv(file_path).head(20)

            # Get AI-driven recommendations from LLM
            llm_recommendations = llm_service.get_recommendations(analysis['summary'], df_sample)
            
            # Attach LLM recommendations to analysis response
            analysis['llm_recommendations'] = llm_recommendations

            results[file.filename] = analysis
        
        print(f"Successfully processed {len(files)} files")
        return UploadResponse(message="Files uploaded successfully", files=results)

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Upload error: {e}")
        raise DatasetError(f"Failed to process uploaded files: {e}")


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(
    files: List[UploadFile] = File(...),
    missing_strategy: str = Form(...),
    scaling: bool = Form(...),
    encoding: str = Form(...),
    target_column: str = Form(None),
    selected_features_json: str = Form(None)
):
    try:
        validate_preprocessing_params(missing_strategy, encoding, target_column)
        selected_features_dict = json.loads(selected_features_json) if selected_features_json else {}
        results = {}
        for file in files:
            print(f"Preprocessing file: {file.filename}")
            await file_validator.validate_file(file)
            file_path = await file_service.save_uploaded_file(file)
            selected_features = selected_features_dict.get(file.filename, None)
            preprocessed_path = preprocessing_service.preprocess_dataset(
                file_path=file_path,
                missing_strategy=missing_strategy,
                scaling=scaling,
                encoding=encoding,
                target_column=target_column,
                selected_features=selected_features
            )
            results[file.filename] = {"preprocessed_file": preprocessed_path}
        print(f"Successfully preprocessed {len(files)} files")
        return PreprocessResponse(message="Preprocessing completed", files=results)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}")


@app.post("/train", response_model=TrainResponse)
async def train_models(
    files: List[str] = Form(...),  # Changed var name for clarity
    target_column: str = Form(None),
    task_type: str = Form(...),
    model_type: str = Form(None),
    tune_hyperparams: bool = Form(False)
):
    """
    Trigger model training. Currently async with Celery, can be adapted to sync as needed.
    """
    results = {}
    try:
        target_columns: Dict[str, str] = json.loads(target_column) if target_column and isinstance(target_column, str) else {}

        for file_path in files:
            print(f"Scheduling training for file: {file_path}")
            if not Path(file_path).exists():
                raise ModelTrainingError(f"File not found: {file_path}")

            # Extract filename to map to target column
            filename = Path(file_path).name.replace("preprocessed_", "")
            col = target_columns.get(filename, target_column if isinstance(target_column, str) else None)

            # Schedule training (will use Celery task as before)
            task = train_model_task.delay(file_path, task_type, model_type, col, None, tune_hyperparams)
            results[file_path] = {"task_id": task.id}
        
        print(f"Training scheduled for {len(files)} files")
        return TrainResponse(message="Training scheduled", results=results)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Training error: {e}")
        raise ModelTrainingError(f"Training scheduling failed: {e}")


@app.post("/predict/{model_id}")
async def predict(model_id: str, data: Dict[str, Any]):
    try:
        model_path = Path(settings.upload_directory) / f"trained_model_{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        result = await async_ml_service.predict_async(model_path, data)
        print(f"Prediction made for model: {model_id}")
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Prediction error: {e}")
        raise ModelTrainingError(f"Prediction failed: {e}")


@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    try:
        model_path = Path(settings.upload_directory) / f"trained_model_{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        print(f"Downloading model: {model_id}")
        return FileResponse(
            path=str(model_path),
            filename=model_path.name,
            media_type="application/octet-stream"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")


@app.get("/health")
async def health_check():
    print("Health check performed")
    return {
        "status": "healthy",
        "message": "No-Code ML Platform API is running",
        "version": "1.0"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
