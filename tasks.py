# Removed Celery and replaced with direct function call for training

from services.ml_service import MLService

def train_model_task(
    file_path,
    task_type,
    model_type,
    target_column,
    user_id=None,
    tune_hyperparams=False
):
    ml_service = MLService()
    return ml_service.train_model(file_path, task_type, model_type, target_column, user_id, tune_hyperparams)
