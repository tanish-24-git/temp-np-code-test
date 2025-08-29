import pytest
from services.ml_service import MLService
import pandas as pd
from pathlib import Path

@pytest.fixture
def ml_service():
    return MLService()

def test_train_classification(ml_service, tmp_path):
    # Create sample data
    data = {
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    
    result = ml_service.train_model(str(file_path), "classification", "logistic_regression", "target")
    assert "results" in result
    assert "accuracy" in result["results"]
    assert result["error"] is None  # Assuming no error

# Add more tests for other methods