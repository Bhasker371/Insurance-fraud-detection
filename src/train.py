"""
Train and compare fraud-detection models, tune the decision threshold, and
save the best model as a reusable bundle.

Pipeline:
    1. Load the joined claims data from SQLite (src/database.py).
    2. Build an sklearn preprocessing + model Pipeline. Feature columns are
       auto-detected (numeric vs categorical), so real data with different
       columns works without code changes.
    3. Train four models, logging params/metrics/artifacts to MLflow.
    4. Pick the best by macro-F1, then tune the probability threshold on a
       validation split (fraud detection cares about recall, so the default
       0.5 cutoff is rarely optimal).
    5. Save {pipeline, threshold, feature_columns, target} to
       models/final_model.joblib for the app and predict.py.

MLflow targets a remote server (e.g. DagsHub) via env vars, falling back to
a local SQLite store so training runs fully offline:
    MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from database import load_features

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "final_model.joblib"
TARGET = "fraud_reported"
ID_COLS = ["policy_id", "claim_id"]


def build_pipeline(numeric, categorical, model) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def tune_threshold(y_true, y_proba) -> float:
    """Return the probability threshold that maximises F1 for the fraud class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[int(np.argmax(f1))])


def configure_tracking():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        local = f"sqlite:///{BASE / 'mlflow.db'}"
        mlflow.set_tracking_uri(local)
        print(f"MLFLOW_TRACKING_URI not set -- using local store {local}")
    mlflow.set_experiment("insurance-claim-fraud")


def main():
    configure_tracking()

    df = load_features()
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET].astype(int)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
    }

    best = {"name": None, "pipeline": None, "f1": -1.0, "auc": 0.0}

    for name, model in candidates.items():
        pipe = build_pipeline(numeric, categorical, model)
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            macro_f1 = f1_score(y_test, pred, average="macro")
            auc = roc_auc_score(y_test, proba)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("macro_f1", macro_f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(pipe, name="model")
            print(f"{name:>24}: macro_f1={macro_f1:.3f}  roc_auc={auc:.3f}")

        if macro_f1 > best["f1"]:
            best = {"name": name, "pipeline": pipe, "f1": macro_f1, "auc": auc}

    # Threshold tuning on a fresh split of the training data.
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.25, random_state=7, stratify=y_train
    )
    best["pipeline"].fit(Xtr, ytr)
    val_proba = best["pipeline"].predict_proba(Xval)[:, 1]
    threshold = tune_threshold(yval, val_proba)

    # Refit the chosen model on all training data before saving.
    best["pipeline"].fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "threshold": threshold,
            "feature_columns": feature_cols,
            "target": TARGET,
        },
        MODEL_PATH,
    )
    print(
        f"\nBest model: {best['name']} (macro_f1={best['f1']:.3f}, "
        f"roc_auc={best['auc']:.3f}), tuned threshold={threshold:.3f}"
    )
    print(f"Saved bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
