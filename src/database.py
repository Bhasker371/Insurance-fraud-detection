"""
Load the claims CSV into a normalized SQLite database and read it back via
a SQL join.

The single wide CSV is split into two related tables that mirror a simple
3NF design -- policy-level attributes in `policies`, claim/incident-level
attributes in `claims` -- linked by `policy_id`. `load_features()` joins
them back into one DataFrame for modelling, demonstrating the SQL layer
without forcing the rest of the pipeline to care about it.
"""

import sqlite3
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "claims.csv"
DB_PATH = BASE / "data" / "insurance.db"

POLICY_COLS = [
    "policy_id", "customer_age", "tenure_months", "policy_type",
    "annual_premium", "region",
]


def build_database(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    """Read the CSV and write normalized `policies` and `claims` tables."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python src/generate_data.py` first "
            "(or drop your own claims.csv there)."
        )

    df = pd.read_csv(csv_path)
    # Only split out columns that are actually present, so a real dataset
    # with a different schema still loads (unknown columns go to `claims`).
    policy_cols = [c for c in POLICY_COLS if c in df.columns]
    claim_cols = [c for c in df.columns if c not in policy_cols or c == "policy_id"]

    policies = df[policy_cols].drop_duplicates(subset="policy_id")
    claims = df[claim_cols].copy()
    claims.insert(0, "claim_id", range(1, len(claims) + 1))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        policies.to_sql("policies", conn, if_exists="replace", index=False)
        claims.to_sql("claims", conn, if_exists="replace", index=False)
    print(f"Built {db_path}  (policies: {len(policies):,}, claims: {len(claims):,})")


def load_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Join policies and claims back into a single modelling DataFrame."""
    if not db_path.exists():
        build_database()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT c.*, p.customer_age, p.tenure_months, p.policy_type,
                   p.annual_premium, p.region
            FROM claims c
            JOIN policies p ON c.policy_id = p.policy_id
            """,
            conn,
        )


if __name__ == "__main__":
    build_database()
