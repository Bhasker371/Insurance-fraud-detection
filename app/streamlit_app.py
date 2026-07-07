"""
Streamlit UI for the insurance-claim fraud detector.

Two modes:
  - Manual input: score a single claim entered in a form.
  - CSV upload: score a batch and download the results.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable when run as `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import load_bundle, predict_df, predict_one  # noqa: E402

st.set_page_config(page_title="Insurance Fraud Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Insurance Claim Fraud Detector")
st.caption(
    "Predicts the probability that an insurance claim is fraudulent, using a "
    "model trained with MLflow experiment tracking and a business-tuned "
    "decision threshold."
)

try:
    bundle = load_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.info(f"Model decision threshold: **{bundle['threshold']:.3f}** "
        "(a claim is flagged as fraud at or above this probability).")

tab_manual, tab_csv = st.tabs(["🔹 Manual input", "🔹 CSV upload (batch)"])


def risk_gauge(probability, is_fraud):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Fraud Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if is_fraud else "darkgreen"},
            "steps": [
                {"range": [0, 33], "color": "#d4edda"},
                {"range": [33, 66], "color": "#fff3cd"},
                {"range": [66, 100], "color": "#f8d7da"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    return fig


with tab_manual:
    with st.form("claim_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_age = st.number_input("Customer age", 18, 90, 40)
            tenure_months = st.number_input("Policy tenure (months)", 1, 240, 36)
            policy_type = st.selectbox("Policy type", ["Auto", "Home", "Health", "Life", "Travel"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        with c2:
            annual_premium = st.number_input("Annual premium", 100.0, 10000.0, 1200.0)
            claim_amount = st.number_input("Claim amount", 0.0, 100000.0, 3000.0)
            incident_type = st.selectbox(
                "Incident type",
                ["Collision", "Theft", "Fire", "Water Damage", "Injury", "Vandalism"],
            )
            incident_severity = st.selectbox(
                "Incident severity", ["Minor", "Moderate", "Major", "Total Loss"]
            )
        with c3:
            incident_hour = st.slider("Incident hour", 0, 23, 14)
            days_to_claim = st.number_input("Days to file claim", 0, 60, 5)
            num_prior_claims = st.number_input("Prior claims", 0, 20, 0)
            witnesses = st.number_input("Witnesses", 0, 10, 1)
            police_report_filed = st.selectbox("Police report filed", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Assess claim")

    if submitted:
        record = {
            "customer_age": customer_age, "tenure_months": tenure_months,
            "policy_type": policy_type, "annual_premium": annual_premium,
            "region": region, "incident_type": incident_type,
            "incident_severity": incident_severity, "incident_hour": incident_hour,
            "days_to_claim": days_to_claim, "num_prior_claims": num_prior_claims,
            "witnesses": witnesses, "police_report_filed": police_report_filed,
            "claim_amount": claim_amount,
            "claim_to_premium_ratio": round(claim_amount / max(annual_premium, 1), 3),
        }
        result = predict_one(record, bundle)
        is_fraud = result["fraud_prediction"] == 1

        if is_fraud:
            st.error("⚠️ This claim is flagged as **LIKELY FRAUD**")
        else:
            st.success("✅ This claim looks **legitimate**")
        st.plotly_chart(risk_gauge(result["fraud_probability"], is_fraud),
                        use_container_width=True)


with tab_csv:
    st.write("Upload a CSV of claims with the same feature columns used in training.")
    uploaded = st.file_uploader("Choose a CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        scored = predict_df(df, bundle)
        flagged = int(scored["fraud_prediction"].sum())
        st.write(f"Scored **{len(scored):,}** claims — flagged **{flagged:,}** as fraud.")
        st.dataframe(scored.head(50), use_container_width=True)
        st.download_button(
            "⬇️ Download results",
            scored.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Trained with scikit-learn + MLflow · Synthetic demo data (swap in real data any time)")
