"""
Generate a synthetic insurance-claims dataset for the fraud-detection
pipeline.

The data is fully synthetic (no real people or policies). Fraud is not
random: it is driven by a latent risk score built from features that are
plausible fraud signals in the real world -- an unusually high claim-to-
premium ratio, no police report, short policy tenure, many prior claims,
odd-hour incidents, and very fast claim filing -- plus noise, so the
models have a genuine but non-trivial signal to learn.

Swapping in real data later: replace data/claims.csv with a real file
that has the same target column (`fraud_reported`, 0/1). The rest of the
pipeline (database, features, training, app) adapts to whatever feature
columns are present.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "claims.csv"

POLICY_TYPES = ["Auto", "Home", "Health", "Life", "Travel"]
INCIDENT_TYPES = ["Collision", "Theft", "Fire", "Water Damage", "Injury", "Vandalism"]
SEVERITIES = ["Minor", "Moderate", "Major", "Total Loss"]
REGIONS = ["North", "South", "East", "West", "Central"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_age = rng.integers(18, 90, n_rows)
    tenure_months = rng.integers(1, 240, n_rows)
    policy_type = rng.choice(POLICY_TYPES, n_rows)
    annual_premium = np.round(rng.uniform(300, 5000, n_rows), 2)
    region = rng.choice(REGIONS, n_rows)

    incident_type = rng.choice(INCIDENT_TYPES, n_rows)
    severity = rng.choice(SEVERITIES, n_rows, p=[0.45, 0.30, 0.18, 0.07])
    incident_hour = rng.integers(0, 24, n_rows)
    days_to_claim = rng.integers(0, 60, n_rows)
    num_prior_claims = rng.poisson(0.8, n_rows)
    witnesses = rng.integers(0, 5, n_rows)
    police_report = rng.choice(["Yes", "No"], n_rows, p=[0.6, 0.4])

    severity_mult = pd.Series(severity).map(
        {"Minor": 0.4, "Moderate": 1.0, "Major": 2.2, "Total Loss": 4.0}
    ).to_numpy()
    claim_amount = np.round(
        annual_premium * severity_mult * rng.uniform(0.3, 2.5, n_rows), 2
    )
    claim_to_premium = claim_amount / annual_premium

    # Latent fraud risk: higher for suspicious combinations.
    risk = (
        -3.2
        + 0.35 * claim_to_premium
        + 0.9 * (police_report == "No")
        + 0.5 * (num_prior_claims >= 2)
        + 0.6 * (tenure_months < 12)
        + 0.4 * ((incident_hour < 5) | (incident_hour >= 23))
        + 0.5 * (days_to_claim <= 1)
        + 0.3 * (witnesses == 0)
        + rng.normal(0, 0.6, n_rows)
    )
    fraud_prob = _sigmoid(risk)
    fraud_reported = (rng.uniform(0, 1, n_rows) < fraud_prob).astype(int)

    return pd.DataFrame({
        "policy_id": [f"POL{100000 + i}" for i in range(n_rows)],
        "customer_age": customer_age,
        "tenure_months": tenure_months,
        "policy_type": policy_type,
        "annual_premium": annual_premium,
        "region": region,
        "incident_type": incident_type,
        "incident_severity": severity,
        "incident_hour": incident_hour,
        "days_to_claim": days_to_claim,
        "num_prior_claims": num_prior_claims,
        "witnesses": witnesses,
        "police_report_filed": police_report,
        "claim_amount": claim_amount,
        "claim_to_premium_ratio": np.round(claim_to_premium, 3),
        "fraud_reported": fraud_reported,
    })


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic claims data.")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    rate = df["fraud_reported"].mean()
    print(f"Wrote {len(df):,} rows to {DATA_PATH}  (fraud rate {rate:.1%})")


if __name__ == "__main__":
    main()
