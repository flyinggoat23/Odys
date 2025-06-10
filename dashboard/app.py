import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("📊 Credit Risk & Provision Dashboard")
st.markdown("Upload your ECL output to explore portfolio risk and provisioning exposure.")

uploaded_file = st.file_uploader("Upload ECL CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    with st.sidebar:
        st.subheader("🔍 Filters")
        stages = st.multiselect("Select Stages", options=sorted(df["Stage"].unique()), default=df["Stage"].unique())
        df = df[df["Stage"].isin(stages)]

        min_pd, max_pd = st.slider("PD Score Range", float(df["pd_score"].min()), float(df["pd_score"].max()), (0.0, 1.0))
        df = df[df["pd_score"].between(min_pd, max_pd)]

        product_types = df["product_type"].unique() if "product_type" in df.columns else []
        if len(product_types) > 0:
            selected_products = st.multiselect("Product Types", product_types, default=product_types)
            df = df[df["product_type"].isin(selected_products)]

    st.subheader("📌 Portfolio Summary")
    st.metric("Total ECL (£)", f"{df['ECL'].sum():,.2f}")
    st.metric("Total Exposure (£)", f"{df['EAD'].sum():,.2f}")
    st.metric("Average PD", f"{df['pd_score'].mean():.3%}")

    st.divider()

    st.subheader("📋 Loan-Level Table")
    st.dataframe(df.sort_values("ECL", ascending=False).reset_index(drop=True))

    st.download_button("📥 Download Filtered Table", df.to_csv(index=False), file_name="filtered_ecl.csv")

    # Moved scenario simulator inside the if block
    st.divider()
    st.subheader("🌀 Scenario Simulator")
    st.markdown("Tweak macro assumptions and see how provisions shift.")

    col1, col2 = st.columns(2)

    with col1:
        pd_shift = st.slider("📈 Increase PD by (%)", -50, 200, 0, step=10)
    with col2:
        lgd_shift = st.slider("📉 Adjust LGD by (%)", -50, 100, 0, step=5)

    adjusted_df = df.copy()
    adjusted_df["adjusted_pd"] = (adjusted_df["pd_score"] * (1 + pd_shift / 100)).clip(0, 1)
    adjusted_df["adjusted_lgd"] = (adjusted_df["LGD"] * (1 + lgd_shift / 100)).clip(0, 1)
    adjusted_df["adjusted_ecl"] = (
        adjusted_df["adjusted_pd"] * adjusted_df["adjusted_lgd"] * adjusted_df["EAD"] * adjusted_df["discount_factor"]
    )

    st.metric("📊 Adjusted ECL (£)", f"{adjusted_df['adjusted_ecl'].sum():,.2f}")
    st.metric("🆚 Change from Base (£)", f"{adjusted_df['adjusted_ecl'].sum() - df['ECL'].sum():,.2f}")

    with st.expander("📋 View Adjusted Loan Table"):
        st.dataframe(adjusted_df[["loan_id", "Stage", "adjusted_pd", "adjusted_lgd", "adjusted_ecl"]].sort_values("adjusted_ecl", ascending=False))

    st.download_button("📥 Download Adjusted Scenario", adjusted_df.to_csv(index=False), file_name="scenario_ecl.csv")

# Show message when no file is uploaded
else:
    st.info("ℹ️ Please upload a CSV file to begin analysis")