from celery import Celery
from config.settings import settings
from services.ml_service import MLService

app = Celery('tasks', broker=settings.celery_broker_url, backend=settings.celery_result_backend)

@app.task
def train_model_task(file_path, task_type, model_type, target_column, user_id, tune_hyperparams):
    ml_service = MLService()
    return ml_service.train_model(file_path, task_type, model_type, target_column, user_id, tune_hyperparams)