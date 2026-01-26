import pytest
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder
from src.models.predict import FraudPredictor
from fastapi.testclient import TestClient
from src.api.main import app


IDENTITY_MODEL_PATH = "artifacts/lightgbm_identity_final.json"
IDENTITY_SCHEMA_PATH = "artifacts/schema/schema_identity_final.json"
STATS_PATH = "artifacts/schema/reference_stats.json"
OPTIMAL_THRESHOLD = 0.474

#Feature builder tests
@pytest.fixture
def builder():
    return FeatureBuilder(schema_path=IDENTITY_SCHEMA_PATH, stats_path=STATS_PATH)

def test_feature_builder_robustness_ratios(builder):
    """Test that interaction features (ratios) are calculated correctly."""
    raw_data = pd.DataFrame([{
        "TransactionAmt": 1000, 
        "DeviceInfo": "Windows", 
        "id_31": "chrome 63.0"
    }])
    processed = builder.transform(raw_data)

    # Assertions for the new Robust features
    assert "Device_Amt_Ratio" in processed.columns
    assert "Browser_Max_Amt" in processed.columns
    assert processed["Device_Amt_Ratio"].iloc[0] > 0

#Prediction Tests
@pytest.fixture
def predictor():
    return FraudPredictor(
        model_path=IDENTITY_MODEL_PATH,
        schema_path=IDENTITY_SCHEMA_PATH,
        stats_path=STATS_PATH
    )

def test_prediction_output_format(predictor):
    """Verify single-record output format (probability as float, not list)."""
    df = pd.DataFrame([{
        "TransactionDT": 86400, 
        "TransactionAmt": 100, 
        "id_31": "chrome 63.0"
    }])
    result = predictor.predict(df)
    
    assert "prediction" in result
    assert "probability" in result
    assert "action" in result
    assert isinstance(result["probability"], float)  # No longer a list
    assert 0 <= result["probability"] <= 1

# API Contract Tests
client = TestClient(app)

def test_api_robustness_action():
    """Verify API returns the correct 'action' string for high-value outliers."""
    payload = {
        "records": [{
            "TransactionAmt": 35000000.0, 
            "id_31": "chrome 63.0",
            "DeviceInfo": "Windows",
            "P_emaildomain": "gmail.com"
        }]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["action"] in ["BLOCK_AND_CHALLENGE", "MANUAL_REVIEW_REQUIRED"]

def test_health_endpoint():
    """Verifying the API is alive."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Robustness & Garbage Handling
@pytest.mark.parametrize("garbage_input", [
    {"TransactionAmt": "invalid_string"}, 
    {"DeviceInfo": None},
    {"id_31": 12345}
])

def test_predictor_garbage_resilience(predictor, garbage_input):
    """Ensure the system handles malformed data without crashing."""
    df = pd.DataFrame([garbage_input])
    try:
        result = predictor.predict(df)
        assert "action" in result
    except Exception as e:
        pytest.fail(f"System crashed on garbage input: {e}")