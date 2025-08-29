import pandas as pd
import numpy as np
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import joblib
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from utils.exceptions import ModelTrainingError

class MLService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
        self._setup_models()
    
    def _setup_models(self):
        self.classification_models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'decision_tree': DecisionTreeClassifier,
            'knn': KNeighborsClassifier,
            'svm': SVC,
        }
        self.regression_models = {
            'linear_regression': LinearRegression,
            'random_forest': RandomForestRegressor,
            'decision_tree': DecisionTreeRegressor,
            'knn': KNeighborsRegressor,
            'svm': SVR,
        }
        self.clustering_models = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
        }
    
    def train_model(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None,
        tune_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """Train a machine learning model with optional hyperparameter tuning"""
        try:
            start_time = time.time()
            df = pd.read_csv(file_path)
            if not model_type:
                model_type = self._get_default_model(task_type)
            if task_type == "classification":
                if model_type not in self.classification_models:
                    raise ValueError(f"Unsupported classification model: {model_type}")
                model_class = self.classification_models[model_type]
            elif task_type == "regression":
                if model_type not in self.regression_models:
                    raise ValueError(f"Unsupported regression model: {model_type}")
                model_class = self.regression_models[model_type]
            elif task_type == "clustering":
                if model_type not in self.clustering_models:
                    raise ValueError(f"Unsupported clustering model: {model_type}")
                model_class = self.clustering_models[model_type]
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            params = settings.default_hyperparameters.get(model_type, {}) if tune_hyperparams else {}
            if task_type == "classification":
                result = self._train_supervised(df, model_class, target_column, task_type, params)
            elif task_type == "regression":
                result = self._train_supervised(df, model_class, target_column, task_type, params)
            elif task_type == "clustering":
                result = self._train_clustering(df, model_class, params)
            
            result["training_time"] = time.time() - start_time
            result["hyperparameters"] = result.get("best_params", {})
            
            model_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            model_filename = f"trained_model_{model_id}_{timestamp}.pkl"
            model_path = self.upload_dir / model_filename
            joblib.dump(result["model"], model_path)
            
            metadata = {
                "model_id": model_id,
                "task_type": task_type,
                "model_type": model_type,
                "metrics": result["results"],
                "hyperparameters": result["hyperparameters"],
                "training_time": result["training_time"],
                "created_at": datetime.utcnow().isoformat()
            }
            metadata_path = self.upload_dir / f"model_metadata_{model_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            del result["model"]
            print(f"Model trained: {model_id}")
            return {"model_id": model_id, **result}
        except Exception as e:
            print(f"Training error: {str(e)}")
            return {
                "task_type": task_type,
                "model_type": model_type,
                "results": {},
                "error": f"Training failed: {str(e)}"
            }
    
    def _train_supervised(self, df: pd.DataFrame, model_class, target_column: str, task_type: str, params: dict) -> Dict[str, Any]:
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.default_test_size, random_state=settings.random_state
        )
        if params:
            grid_search = GridSearchCV(model_class(), params, cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model = model_class()
            model.fit(X_train, y_train)
            best_params = {}
        
        y_pred = model.predict(X_test)
        if task_type == "classification":
            results = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:  # regression
            results = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
                "mean_absolute_error": float(mean_absolute_error(y_test, y_pred))
            }
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        except Exception as e:
            print(f"Cross-validation failed: {str(e)}")
        
        feature_importance = self._get_feature_importance(model, X.columns)
        return {
            "task_type": task_type,
            "model_type": model_class.__name__.lower(),
            "results": results,
            "feature_importance": feature_importance,
            "best_params": best_params,
            "model": model
        }
    
    def _train_clustering(self, df: pd.DataFrame, model_class, params: dict) -> Dict[str, Any]:
        X = df.select_dtypes(include=[np.number])
        if X.empty:
            raise ValueError("No numeric columns found for clustering")
        if params:
            grid_search = GridSearchCV(model_class(), params, cv=5, scoring='silhouette_score')
            grid_search.fit(X)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model = model_class()
            model.fit(X)
            best_params = {}
        
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        results = {}
        if len(np.unique(labels)) > 1 and -1 not in labels:
            try:
                results["silhouette_score"] = float(silhouette_score(X, labels))
                results["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
            except Exception as e:
                print(f"Clustering metrics failed: {str(e)}")
        return {
            "task_type": "clustering",
            "model_type": model_class.__name__.lower(),
            "results": results,
            "best_params": best_params,
            "model": model
        }
    
    def _get_default_model(self, task_type: str) -> str:
        defaults = {
            "classification": "logistic_regression",
            "regression": "linear_regression",
            "clustering": "kmeans"
        }
        return defaults.get(task_type, "logistic_regression")
    
    def _get_feature_importance(self, model, feature_names) -> Optional[List[List[Any]]]:
        try:
            importance_dict = {}
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
                else:
                    importance_dict = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
            if importance_dict:
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return sorted_importance[:10]
            return None
        except Exception as e:
            print(f"Feature importance extraction failed: {str(e)}")
            return None

class AsyncMLService:
    def __init__(self):
        self.ml_service = MLService()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def train_model_async(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None,
        tune_hyperparams: bool = False
    ) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.ml_service.train_model,
                file_path,
                task_type,
                model_type,
                target_column,
                tune_hyperparams
            )
            return result
        except Exception as e:
            print(f"Async training error: {str(e)}")
            raise ModelTrainingError(f"Async training failed: {str(e)}")
    
    async def predict_async(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                model_path,
                data
            )
            return result
        except Exception as e:
            print(f"Async prediction error: {str(e)}")
            raise ModelTrainingError(f"Async prediction failed: {str(e)}")
    
    def _predict_sync(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        model = joblib.load(model_path)
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df).tolist()
        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities,
            "model_id": model_path.stem.split('_')[2]
        }