"""
Load the saved model bundle and score insurance claims.

Used by the Streamlit app and usable on its own:
    from predict import load_bundle, predict_df
    bundle = load_bundle()
    result = predict_df(df, bundle)   # df of claim features
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model.joblib"


@lru_cache(maxsize=1)
def load_bundle(model_path: str = str(MODEL_PATH)) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python src/train.py`."
        )
    return joblib.load(path)


def predict_df(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """
    Return a copy of `df` with `fraud_probability` and `fraud_prediction`
    columns. Missing feature columns are filled with NA so the trained
    pipeline's encoding can handle partial input.
    """
    bundle = bundle or load_bundle()
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_columns"]

    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[feature_cols]

    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba.round(4)
    out["fraud_prediction"] = (proba >= threshold).astype(int)
    return out


def predict_one(record: dict, bundle: dict | None = None) -> dict:
    """Score a single claim given as a dict of feature -> value."""
    result = predict_df(pd.DataFrame([record]), bundle)
    return {
        "fraud_probability": float(result["fraud_probability"].iloc[0]),
        "fraud_prediction": int(result["fraud_prediction"].iloc[0]),
    }
