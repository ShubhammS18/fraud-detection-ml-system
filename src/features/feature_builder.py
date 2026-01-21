# src/feature_builder.py

import numpy as np
import pandas as pd
from pathlib import Path
import json

SECONDS_IN_HOUR = 3600


def build_features(df: pd.DataFrame,target_col: str | None = None,is_training: bool = False):
    """
    Build model-ready features from raw transaction data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    target_col : str | None
        Name of target column (e.g. 'isFraud') if present.
    is_training : bool
        Whether this is training mode (returns y).

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series | None
        Target series (training only).
    """

    df = df.copy()

    # Ensuring TransactionAmt is numeric before math, or it crashes on strings
    if "TransactionAmt" in df.columns:
        df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors='coerce').fillna(0)
    
    if "TransactionDT" in df.columns:
        df["TransactionDT"] = pd.to_numeric(df["TransactionDT"], errors='coerce').fillna(0)

    y = None
    if is_training and target_col is not None:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    # Time-based feature
    if "TransactionDT" in df.columns:
        df["transaction_hour"] = (df["TransactionDT"] // SECONDS_IN_HOUR) % 24

    # Transaction amount transformation
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df.drop(columns=["TransactionAmt"], inplace=True)

    
    # Categorical missing value handling
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")


    # Final validation
    if df.isnull().sum().sum() != 0:
        raise ValueError("NaNs detected after feature engineering")

    return (df, y) if is_training else df


class FeatureBuilder:
    def __init__(self, schema_path: str):
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")

        with open(schema_path) as f:
            schema = json.load(f)

        self.features = schema["features"]
        self.categorical_features = set(schema["categorical_features"])


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = build_features(df, is_training=False)

        # Ensuring all required features exist
        for col in self.features:
            if col not in X.columns:
                # If it's one of the 13 categorical features, use 'Unknown'
                if col in self.categorical_features:
                    X[col] = "Unknown"
                else:
                    X[col] = 0

        # Strict alignment with the schema features list
        X = X[self.features].copy()
        return X
