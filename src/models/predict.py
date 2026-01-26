import lightgbm as lgb
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder
from pathlib import Path
import json
class FraudPredictor:
    def __init__(self, model_path: str, schema_path: str, stats_path: str = "artifacts/schema/reference_stats.json"):
        self.model = lgb.Booster(model_file=str(model_path))
        self.feature_builder = FeatureBuilder(schema_path=str(schema_path), stats_path=stats_path)
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
            
        self.model_features = self.model.feature_name()
        
        self.cat_cols = self.schema.get("categorical_features", [])
        self.vocab = self.schema.get("category_vocabulary", {})

    def predict(self, df: pd.DataFrame):
        # 1. Feature Engineering
        X = self.feature_builder.transform(df)

        # 2. Strict Column Alignment
        X = X[self.model_features].copy()

        # 3. TYPING ENFORCEMENT
        for col in self.model_features:
            if col in self.cat_cols:
                # Map strings to the exact integer indices the model expects
                v = self.vocab.get(col, [])
                if v:
                    X[col] = pd.Categorical(X[col], categories=v)
                else:
                    X[col] = X[col].astype('category')
            else:
                # Force numeric for all other features (including new Ratios)
                X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32').fillna(0)
        
        # 4. Final Prediction & Action Logic
        proba = self.model.predict(X)[0]
        
        #DEFENSIVE DATA EXTRACTION (Fixes Garbage Input Crashes)
        raw_amt = df.get('TransactionAmt', pd.Series([0])).iloc[0]
        try:
            amt = float(raw_amt)
        except (ValueError, TypeError):
            amt = 0.0
        
        # Robust Action Logic: Hybrid of ML + Risk Thresholds
        OPTIMAL_THRESHOLD = 0.474
        if proba >= OPTIMAL_THRESHOLD:
            action = "BLOCK_AND_CHALLENGE"
        elif amt > 100000:
            # If it's a huge amount and even slightly suspicious, flag for review
            action = "MANUAL_REVIEW_REQUIRED"
        elif amt > 2000 and proba > 0.10:
            action = "MANUAL_REVIEW_REQUIRED"
        else:
            action = "APPROVE"
            
        return {
            "prediction": 1 if action == "BLOCK_AND_CHALLENGE" else 0,
            "probability": float(proba),
            "action": action }