import os
import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from models.requests import PreprocessRequest, TrainRequest
from models.responses import UploadResponse, PreprocessResponse, TrainResponse
from services.file_service import FileService
from services.preprocessing_service import PreprocessingService
from services.ml_service import MLService, AsyncMLService
from utils.validators import file_validator, validate_preprocessing_params
from utils.exceptions import DatasetError, ModelTrainingError, PreprocessingError, ValidationError
from tasks import train_model_task  # Celery task

# Initialize FastAPI app
app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for training ML models without code",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
file_service = FileService()
preprocessing_service = PreprocessingService()
ml_service = MLService()
async_ml_service = AsyncMLService()

# Ensure upload directory exists
os.makedirs(settings.upload_directory, exist_ok=True)

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and analyze datasets"""
    try:
        results = {}
        for file in files:
            print(f"Processing file: {file.filename}")
            await file_validator.validate_file(file)
            file_path = await file_service.save_uploaded_file(file)
            analysis = file_service.analyze_dataset(file_path)
            results[file.filename] = analysis
        print(f"Successfully processed {len(files)} files")
        return UploadResponse(message="Files uploaded successfully", files=results)
    except Exception as e:
        print(f"Upload error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise DatasetError(f"Failed to process uploaded files: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(
    files: List[UploadFile] = File(...),
    missing_strategy: str = Form(...),
    scaling: bool = Form(...),
    encoding: str = Form(...),
    target_column: str = Form(None),
    selected_features_json: str = Form(None)
):
    """Preprocess uploaded datasets"""
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
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        if isinstance(e, (HTTPException, ValidationError)):
            raise e
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")

@app.post("/train", response_model=TrainResponse)
async def train_models(
    background_tasks: BackgroundTasks,
    preprocessed_filenames: List[str] = Form(...),
    target_column: str = Form(None),
    task_type: str = Form(...),
    model_type: str = Form(None),
    tune_hyperparams: bool = Form(False)
):
    """Train ML models asynchronously"""
    try:
        target_columns = json.loads(target_column) if target_column and isinstance(target_column, str) else {}
        results = {}
        for preprocessed_file in preprocessed_filenames:
            print(f"Scheduling training for file: {preprocessed_file}")
            if not os.path.exists(preprocessed_file):
                raise ModelTrainingError(f"Preprocessed file not found: {preprocessed_file}")
            filename = os.path.basename(preprocessed_file).replace("preprocessed_", "")
            file_target = target_columns.get(filename, target_column if isinstance(target_column, str) else None)
            task = train_model_task.delay(preprocessed_file, task_type, model_type, file_target, tune_hyperparams)
            results[preprocessed_file] = {"task_id": task.id}
        print(f"Successfully scheduled training for {len(preprocessed_filenames)} files")
        return TrainResponse(message="Training scheduled", results=results)
    except Exception as e:
        print(f"Training error: {str(e)}")
        if isinstance(e, (HTTPException, ModelTrainingError)):
            raise e
        raise ModelTrainingError(f"Training scheduling failed: {str(e)}")

@app.post("/predict/{model_id}")
async def predict(model_id: str, data: Dict[str, Any]):
    """Make predictions using trained model"""
    try:
        model_path = Path(settings.upload_directory) / f"trained_model_{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        result = await async_ml_service.predict_async(model_path, data)
        print(f"Prediction made for model: {model_id}")
        return result
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise ModelTrainingError(f"Prediction failed: {str(e)}")

@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    """Download trained model file"""
    try:
        model_path = Path(settings.upload_directory) / f"trained_model_{model_id}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        print(f"Downloading model: {model_id}")
        return FileResponse(
            path=model_path,
            filename=os.path.basename(model_path),
            media_type='application/octet-stream'
        )
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    print("Health check performed")
    return {
        "status": "healthy",
        "message": "No-Code ML Platform API is running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
