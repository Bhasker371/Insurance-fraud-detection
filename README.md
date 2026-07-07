# 🛡️ Insurance Claim Fraud Detection (MLOps)

An end-to-end machine-learning system that predicts whether an insurance
claim is fraudulent, covering the full workflow: data → normalized SQL
storage → feature engineering → model training with experiment tracking →
threshold tuning → an interactive prediction app.

The pipeline ships with a **synthetic data generator** so it runs
end-to-end out of the box, with no external dataset required. It is built
to accept **real data with zero code changes** — drop a real
`data/claims.csv` (with a `fraud_reported` 0/1 target) in place of the
generated one and re-run the pipeline.

## What it does

1. **Synthetic data** (`src/generate_data.py`) — generates realistic
   insurance claims where fraud is driven by plausible signals (high
   claim-to-premium ratio, no police report, short tenure, prior claims,
   odd-hour incidents, fast filing) plus noise.
2. **Normalized storage** (`src/database.py`) — loads the CSV into a
   SQLite database split into `policies` and `claims` tables (a simple 3NF
   layout), then joins them back for modelling.
3. **Training + tracking** (`src/train.py`) — trains and compares four
   models (Logistic Regression, Random Forest, Gradient Boosting,
   Histogram Gradient Boosting) inside an sklearn `Pipeline`, logging
   params, metrics, and artifacts to **MLflow**.
4. **Threshold tuning** — fraud detection cares about catching fraud
   (recall), so the default 0.5 cutoff is rarely optimal; the decision
   threshold is tuned on a validation split to maximise F1.
5. **Serving** (`src/predict.py`, `app/streamlit_app.py`) — the best model
   + tuned threshold + feature list are saved as a single joblib bundle,
   served through a Streamlit app with **manual single-claim** and
   **CSV batch** modes.

## Experiment tracking

Set these to log to a remote MLflow server (e.g. [DagsHub](https://dagshub.com)):

```
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/insurance-claim-fraud.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-token>
```

Keep these in a local `.env` (gitignored) — never commit them. If unset,
training falls back to a local SQLite MLflow store so it runs fully
offline.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

python src/generate_data.py     # create synthetic data/claims.csv
python src/database.py          # build the SQLite database
python src/train.py             # train, tune, save models/final_model.joblib
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Using real data

Replace `data/claims.csv` with your own claims file that has a
`fraud_reported` column (0 = legitimate, 1 = fraud). The database,
feature detection, training, and app all adapt to the columns present —
no code changes needed. Then re-run the four commands above.

## Project structure

```
insurance-fraud-detection/
├── data/                      # generated CSV + SQLite DB (gitignored)
├── src/
│   ├── generate_data.py       # synthetic claims generator
│   ├── database.py            # CSV -> SQLite (policies + claims) + join
│   ├── train.py               # train/compare models, MLflow, threshold tuning
│   └── predict.py             # load bundle, score single/batch claims
├── app/
│   └── streamlit_app.py       # manual + CSV-batch prediction UI
├── models/                    # final_model.joblib (created by train.py)
├── requirements.txt
├── .gitignore
└── README.md
```
