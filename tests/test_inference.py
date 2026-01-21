import pytest
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder
from src.models.predict import FraudPredictor
from fastapi.testclient import TestClient
from src.api.main import app

#Feature builder tests
@pytest.fixture
def builder():
    # Uses verified v2 schema
    return FeatureBuilder(schema_path="artifacts/schema/lightgbm_v2_schema.json")

def test_feature_builder_missing_columns(builder):
    """Test that missing columns are filled based on schema logic."""
    raw_data = pd.DataFrame([{"TransactionDT": 100}]) # Missing almost everything
    processed = builder.transform(raw_data)
    
    # Assertions
    assert len(processed.columns) == len(builder.features)
    assert processed["ProductCD"].iloc[0] == "Unknown"
    assert processed["card1"].iloc[0] == 0

def test_feature_builder_column_order(builder):
    """Test that output columns exactly match schema order regardless of input order."""
    reversed_cols = [{"card1": 123, "TransactionDT": 86400, "ProductCD": "W"}]
    processed = builder.transform(pd.DataFrame(reversed_cols))
    
    assert list(processed.columns) == builder.features

def test_feature_builder_unknown_categories(builder):
    """Test that new, unseen categorical values don't crash the system."""
    raw_data = pd.DataFrame([{"ProductCD": "Z"}]) # 'Z' was never in training
    processed = builder.transform(raw_data)
    
    assert processed["ProductCD"].iloc[0] == "Z" 
    # Note: Our predict.py handles the mapping of 'Z' to 'category' safely



#Prediction tests 
@pytest.fixture
def predictor():
    return FraudPredictor(
        model_path="artifacts/lightgbm_v2_pruned.json",
        schema_path="artifacts/schema/lightgbm_v2_schema.json"
    )

def test_prediction_output_format(predictor):
    """Verify shape and value ranges of the prediction."""
    df = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 100, "ProductCD": "W"}])
    result = predictor.predict(df)
    
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"][0], int)
    assert 0 <= result["probability"][0] <= 1

def test_batch_prediction_consistency(predictor):
    """Ensure batching multiple records returns the correct number of results."""
    df = pd.DataFrame([{"card1": 1}, {"card1": 2}, {"card1": 3}])
    result = predictor.predict(df)
    
    assert len(result["prediction"]) == 3
    assert len(result["probability"]) == 3
    
    
#API Contract tests 
client = TestClient(app)

def test_health_endpoint():
    """Verify the API is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_api_invalid_payload():
    """Verify the API rejects malformed JSON (Contract Safety)."""
    # Sending a list instead of a dict with 'records' key
    response = client.post("/predict", json=[{"bad": "data"}])
    assert response.status_code == 422 # Unprocessable Entity
    

#mocking the model
from unittest.mock import MagicMock

def test_prediction_mocked(predictor, monkeypatch):
    """Testing logic without loading the heavy model file."""
    # 1. Create a fake model response (mock)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.99]) # Force a high fraud proba
    mock_model.feature_name.return_value = predictor.model_features
    
    # 2. Replace the real model inside predictor with our mock
    monkeypatch.setattr(predictor, "model", mock_model)
    
    # 3. Test the predictor logic
    df = pd.DataFrame([{"TransactionDT": 123}])
    result = predictor.predict(df)
    
    assert result["prediction"] == [1]  # Because 0.99 >= 0.5
    assert result["probability"] == [0.99]
    mock_model.predict.assert_called_once() 
    
    
#Testing with Garbage Input

@pytest.mark.parametrize("garbage_input", [
    {"TransactionAmt": "Extremely Expensive"},  # String instead of float
    {"card1": None},                            # Null instead of int
    {"ProductCD": 99999}                        # Number instead of string
    ])
def test_property_garbage_input(predictor, garbage_input):
    """Ensure 'garbage' data is handled gracefully, not crashing."""
    df = pd.DataFrame([garbage_input])
    
    # Our code should catch this or fill it via FeatureBuilder
    try:
        result = predictor.predict(df)
        assert "prediction" in result
    except Exception as e:
        pytest.fail(f"Predictor crashed on garbage input {garbage_input}: {e}")