import lightgbm as lgb
import pandas as pd
import numpy as np
from src.features.feature_builder import FeatureBuilder
from pathlib import Path

class FraudPredictor:
    def __init__(self, model_path: str, schema_path: str):
        self.model = lgb.Booster(model_file=str(model_path))
        self.feature_builder = FeatureBuilder(schema_path=str(schema_path))
        self.model_features = self.model.feature_name()
        # Getting the 13 categorical names from the builder (loaded from schema JSON)
        self.cat_cols = list(self.feature_builder.categorical_features)

    def predict(self, df: pd.DataFrame):
        # 1. Feature Engineering
        X = self.feature_builder.transform(df)

        # 2. Strict Column Alignment (Order must match model)
        X = X[self.model_features].copy()

        # 3. TYPING ENFORCEMENT
        for col in self.model_features:
            if col in self.cat_cols:
                # Force to category - this satisfies the "dataset mismatch" error
                X[col] = X[col].astype('category')
            else:
                # Force all others to numeric to avoid 'object' type issues
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # 4. Final Prediction
        proba = self.model.predict(X)
        preds = (proba >= 0.5).astype(int)

        return {
            "prediction": preds.tolist(),
            "probability": proba.tolist()
        }