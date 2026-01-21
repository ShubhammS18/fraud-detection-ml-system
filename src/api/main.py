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
    actions: List[str]



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
        
        #Business Logic Thresholding
        actions = []
        for prob in results["probability"]:
            if prob >= 0.50:
                actions.append("BLOCK_AND_CHALLENGE") # e.g. Trigger SMS/EMail OTP
            elif prob >= 0.10:
                actions.append("MANUAL_REVIEW_REQUIRED")
            else:
                actions.append("APPROVE")
        
        results["actions"] = actions
        
        latency = time.time() - start_time
        
        # 4. Logging the Success with metadata (Inference time & Confidence)
        logger.info("Prediction successful", extra={
            "request_id": request_id,
            "latency_ms": round(latency * 1000, 2),
            "avg_probability": sum(results["probability"]) / len(results["probability"]),
            "primary_action": actions[0] if actions else "NONE"
        })
        
        return results

    except Exception as e:
        # 5. Logging the Failure (Crucial for debugging production)
        logger.error(f"Prediction failed", extra={
            "request_id": request_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Inference failed")
