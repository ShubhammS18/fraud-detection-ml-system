from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any

from src.models.predict import FraudPredictor
import time
import uuid
from src.utils.logger import logger


# App initialization
app = FastAPI(
    title="Fraud Detection API",
    version="2.0",
    description="Inference API for Identity-Enhanced LightGBM fraud detection model"
)


# Loading model ONCE
MODEL_PATH = "artifacts/lightgbm_identity_final.json"
SCHEMA_PATH = "artifacts/schema/schema_identity_final.json"
STATS_PATH = "artifacts/schema/reference_stats.json"


try:
    predictor = FraudPredictor(model_path=MODEL_PATH, schema_path=SCHEMA_PATH, stats_path=STATS_PATH)
except Exception as e:
    logger.error(f"Critical System Failure: Model Load Error - {str(e)}")
    raise RuntimeError(f"Failed to load model: {e}")



# Request / Response Schemas
class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    action: str



# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict(request_data: PredictRequest, request: Request):
    # 1. Generating a unique Request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # 2. Logging the incoming request
    logger.info(f"Prediction started", extra={
        "request_id": request_id,
        "record_count": len(request_data.records),
        "endpoint": "/predict"
    })

    try:
        df = pd.DataFrame(request_data.records)
        
        # 3. Prediction
        results = predictor.predict(df)
        
        latency = time.time() - start_time
        
        # 4. Logging the Success with metadata (Inference time & Confidence)
        logger.info("Prediction successful", extra={
            "request_id": request_id,
            "latency_ms": round(latency * 1000, 2),
            "probability": results["probability"],
            "primary_action": results["action"]
        })
        
        return results

    except Exception as e:
        # 5. Logging the Failure (Crucial for debugging production)
        logger.error(f"Prediction failed", extra={
            "request_id": request_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
