import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Provision Calculator", layout="wide")
st.title("Provision Calculator Interface")

# Load PD model
@st.cache_resource
def load_model():
    return joblib.load("pd/pd_model.joblib")

model = load_model()

# Feature list
FEATURES = [
    "loan_id", "loan_amount", "term_months", "interest_rate",
    "missed_payments", "avg_payment_ratio", "payment_volatility", "last_payment_gap"
]

# Sample input guide
sample = pd.DataFrame({
    "loan_id": ["L001", "L002"],
    "loan_amount": [5000, 10000],
    "term_months": [12, 24],
    "interest_rate": [0.08, 0.10],
    "missed_payments": [0, 2],
    "avg_payment_ratio": [1.0, 0.6],
    "payment_volatility": [0.05, 0.2],
    "last_payment_gap": [0, 45]
})

st.markdown("### 📋 Step 1: Paste loan data or upload CSV")


# Paste or upload
tab1, tab2 = st.tabs(["📋 Paste Data", "📂 Upload CSV"])

with tab1:
    input_text = st.text_area("Paste CSV data here (including header)", value=sample.to_csv(index=False), height=200)
    try:
        input_df = pd.read_csv(io.StringIO(input_text))
    except Exception as e:
        st.error(f"Error reading pasted data: {e}")
        input_df = None

with tab2:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            input_df = None



# Optional: Visualize ECL under different scenarios
if all(s in df.columns for s in ["ecl_baseline", "ecl_adverse", "ecl_optimistic"]):
    st.markdown("ECL Under Different Scenarios")
    scenario_summary = df[["ecl_baseline", "ecl_adverse", "ecl_optimistic"]].sum().reset_index()
    scenario_summary.columns = ["Scenario", "ECL"]

    import plotly.express as px
    fig = px.bar(scenario_summary, x="Scenario", y="ECL", text_auto=True,
                 title="Provisioning Impact by Scenario")
    st.plotly_chart(fig, use_container_width=True)


# Proceed if input is valid
if input_df is not None:
    missing_cols = [col for col in FEATURES if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        st.success("✅ Valid input detected. Running provisioning...")

        # Fill missing behavioral defaults
        input_df.fillna({
            "missed_payments": 0,
            "avg_payment_ratio": 1.0,
            "payment_volatility": 0.0,
            "last_payment_gap": 0
        }, inplace=True)

        # Predict PD
        X = input_df[[f for f in FEATURES if f != "loan_id"]]
        input_df["pd"] = model.predict_proba(X)[:, 1]

        # Stage assignment (simplified logic)
        def assign_stage(row):
            if row["missed_payments"] > 3 or row["last_payment_gap"] > 90:
                return 3
            elif row["missed_payments"] > 0 or row["pd"] > 0.3:
                return 2
            else:
                return 1

        input_df["stage"] = input_df.apply(assign_stage, axis=1)

        # ECL calculation
        input_df["ead"] = input_df["loan_amount"]
        input_df["lgd"] = 0.45

        # Macro-weighted ECL
        scenarios = {"baseline": 1.0, "adverse": 1.5, "optimistic": 0.7}
        for name, weight in scenarios.items():
            input_df[f"ecl_{name}"] = input_df["pd"] * weight * input_df["lgd"] * input_df["ead"]

        input_df["ecl"] = (
            0.6 * input_df["ecl_baseline"] +
            0.3 * input_df["ecl_adverse"] +
            0.1 * input_df["ecl_optimistic"]
        )

        # Display
        st.markdown("### 🔍 Loan-Level Results")
        st.dataframe(input_df[["loan_id", "pd", "stage", "ead", "ecl"]])

        # Summary
        st.markdown("### 📊 Summary")
        col1, col2 = st.columns(2)
        col1.metric("Loans", f"{len(input_df):,}")
        col2.metric("Total Provision", f"£{input_df['ecl'].sum():,.0f}")

        # Download
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", csv, "provision_results.csv", "text/csv")

else:
    st.info("Awaiting valid input to begin provisioning.")
