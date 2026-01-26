# src/feature_builder.py

import numpy as np
import pandas as pd
from pathlib import Path
import json

SECONDS_IN_HOUR = 3600

class FeatureBuilder:
    def __init__(self, schema_path: str, stats_path: str = "artifacts/schema/reference_stats.json"):
        schema_path = Path(schema_path)
        stats_path = Path(stats_path)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")

        with open(schema_path) as f:
            self.schema = json.load(f)

        # Load reference statistics for robustness (Ratios)
        if stats_path.exists():
            with open(stats_path) as f:
                self.stats = json.load(f)
        else:
            # Fallback if stats aren't generated yet to prevent crashing
            print(f"⚠️ Warning: Stats not found at {stats_path}. Ratios will default to 0.")
            self.stats = {"device_avg": {}, "browser_max": {}, "email_avg": {}, "global_mean": 100.0}

        self.features = self.schema["features"]
        self.categorical_features = set(self.schema["categorical_features"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw input into robust features including real-time interaction ratios.
        """
        X = df.copy()

        # 1. Ensure numeric types for core columns
        if "TransactionAmt" in X.columns:
            X["TransactionAmt"] = pd.to_numeric(X["TransactionAmt"], errors='coerce').fillna(0)
        
        if "TransactionDT" in X.columns:
            X["TransactionDT"] = pd.to_numeric(X["TransactionDT"], errors='coerce').fillna(0)

        # 2. Time-based feature
        if "TransactionDT" in X.columns:
            X["transaction_hour"] = (X["TransactionDT"] // SECONDS_IN_HOUR) % 24

        # 3. ROBUSTNESS INJECTION: Real-time Interaction Ratios
        if "TransactionAmt" in X.columns:
            amt = X["TransactionAmt"].iloc[0]
            
            # Extract identifiers for stats lookup
            device = str(X.get("DeviceInfo", ["Unknown"])[0]).lower()
            browser = str(X.get("id_31", ["Unknown"])[0]).lower()
            email = str(X.get("P_emaildomain", ["Unknown"])[0]).lower()

            # Lookup training-set means
            dev_mean = self.stats['device_avg'].get(device, self.stats['global_mean'])
            brw_max = self.stats['browser_max'].get(browser, self.stats['global_mean'])
            eml_mean = self.stats['email_avg'].get(email, self.stats['global_mean'])

            # Create the 4 Interaction Features used in training
            X['Device_Avg_Amt'] = dev_mean
            X['Device_Amt_Ratio'] = amt / (dev_mean + 1)
            X['Browser_Max_Amt'] = brw_max
            X['Email_P_Ratio'] = amt / (eml_mean + 1)

            # Create Log Transform
            X["TransactionAmt_log"] = np.log1p(amt)
            
            # Drop raw amount as we did in the notebook
            X.drop(columns=["TransactionAmt"], inplace=True)

        # 4. Handle Categorical Missing Values & Alignment
        # Normalize strings to match training vocabulary (lowercase)
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna("unknown").astype(str).str.lower()

        # 5. Column Alignment & Memory Management
        missing_cols = {}
        for col in self.features:
            if col not in X.columns:
                missing_cols[col] = "unknown" if col in self.categorical_features else 0
        
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=X.index)
            X = pd.concat([X, missing_df], axis=1)

        # Strict order and de-fragmentation
        X = X[self.features].copy() 

        # 6. Ensure Categorical Types (Crucial for LightGBM)
        for col in self.categorical_features:
            X[col] = X[col].astype('category')

        return X