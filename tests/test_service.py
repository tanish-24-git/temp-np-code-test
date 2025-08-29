import pytest
import pandas as pd

from services.ml_service import MLService


@pytest.fixture
def ml_service():
    return MLService()


def test_train_classification(tmp_path, ml_service):
    df = pd.DataFrame({
        "feat1": [1,2,3,4,5],
        "feat2": [5,4,3,2,1],
        "target": [0,1,0,1,0],
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result = ml_service.train_model(str(file_path), "classification", "logistic_regression", "target")
    assert "results" in result
    assert "accuracy" in result["results"]
    assert result.get("error") is None


def test_train_regression(tmp_path, ml_service):
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "target": [2.1, 4.2, 6.3, 8.4, 10.5],
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result = ml_service.train_model(str(file_path), "regression", "linear_regression", "target")
    assert "results" in result
    assert "r2_score" in result["results"]
    assert result.get("error") is None


def test_train_clustering(tmp_path, ml_service):
    df = pd.DataFrame({
        "feat1": [1,1,2,2,3,3],
        "feat2": [1,2,1,2,1,2],
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result = ml_service.train_model(str(file_path), "clustering", "kmeans")
    assert "results" in result
    assert ("silhouette_score" in result["results"]) or ("calinski_harabasz_score" in result["results"])
    assert result.get("error") is None


def test_invalid_target_column(tmp_path, ml_service):
    df = pd.DataFrame({
        "feat1": [1,2,3],
        "feat2": [4,5,6]
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result = ml_service.train_model(str(file_path), "classification", "logistic_regression", "missing_target")
    assert result["error"] is not None
