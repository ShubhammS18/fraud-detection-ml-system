from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any

from src.models.predict import FraudPredictor


# App initialization
app = FastAPI(
    title="Fraud Detection API",
    version="1.0",
    description="Inference API for LightGBM fraud detection model"
)


# Loading model ONCE
MODEL_PATH = "artifacts/lightgbm_v2_pruned.json"
SCHEMA_PATH = "artifacts/schema/lightgbm_v2_schema.json"

try:
    predictor = FraudPredictor(model_path=MODEL_PATH, schema_path=SCHEMA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")



# Request / Response Schemas
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    prediction: List[int]
    probability: List[float]



# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame(request.records)

        if df.empty:
            raise ValueError("Empty input data")

        result = predictor.predict(df)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
