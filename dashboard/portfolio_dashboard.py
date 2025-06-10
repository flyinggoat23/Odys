import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ecl/ecl_results.csv")
    return df

df = load_data()

st.title("📊 Credit Portfolio Risk Dashboard")

# Basic Metrics
total_loans = len(df)
total_exposure = df["ead"].sum()
total_ecl = df["ecl"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Loans", f"{total_loans:,}")
col2.metric("Total Exposure", f"£{total_exposure:,.0f}")
col3.metric("Weighted ECL", f"£{total_ecl:,.0f}")

st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filter Portfolio")

df["loan_amount"] = df["loan_amount"].fillna(0)
min_loan = int(df["loan_amount"].min())
max_loan = int(df["loan_amount"].max())
if min_loan == max_loan:
    st.sidebar.warning(
        "⚠️ Loan amount data is missing or all values are the same. Using default loan range: £0-£100,000"
    )
    min_loan = 0
    max_loan = 100000

min_amt, max_amt = st.sidebar.slider(
    "Loan Amount Range (£)",
    min_value=min_loan,
    max_value=max_loan,
    value=(min_loan, max_loan),
)

term_options = sorted(df["term_months"].dropna().unique())
if not term_options:
    term_options = [12, 24, 36, 48, 60]
    st.sidebar.warning("⚠️ No term data found. Using default terms")

selected_term = st.sidebar.selectbox("Term Length (months)", term_options)

# Stage filter
if "stage" in df.columns:
    stage_options = sorted(df["stage"].dropna().unique())
    selected_stage = st.sidebar.multiselect("IFRS Stage", stage_options, default=stage_options)
else:
    selected_stage = None

# Macro scenario selection
scenarios = ["baseline", "adverse", "optimistic"]
selected_scenario = st.sidebar.selectbox("Select Macro Scenario", scenarios)

# Apply filters
filtered_df = df[
    (df["loan_amount"] >= min_amt)
    & (df["loan_amount"] <= max_amt)
    & (df["term_months"] == selected_term)
]

if selected_stage:
    filtered_df = filtered_df[filtered_df["stage"].isin(selected_stage)]

st.subheader("📌 Filtered Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Loans", f"{len(filtered_df):,}")
col2.metric("Exposure", f"£{filtered_df['ead'].sum():,.0f}")
col3.metric("ECL", f"£{filtered_df[f'ecl_{selected_scenario}'].sum():,.0f}")

# ECL by Stage (filtered)
if "stage" in filtered_df.columns:
    st.markdown("### 🧮 ECL by Stage (Filtered)")
    stage_summary = filtered_df.groupby("stage").agg(
        ecl_total=(f"ecl_{selected_scenario}", "sum"),
        loans=("loan_id", "count"),
    ).reset_index()
    fig2 = px.bar(
        stage_summary,
        x="stage",
        y="ecl_total",
        title=f"ECL by IFRS Stage ({selected_scenario.capitalize()})",
        text="loans",
    )
    st.plotly_chart(fig2, use_container_width=True)

# Raw data toggle
with st.expander("🧾 Show Raw Data"):
    st.dataframe(filtered_df.head(100))
