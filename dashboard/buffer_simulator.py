import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Regulatory Buffer Simulator", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ecl/ecl_results.csv")
    return df

df = load_data()

# Title
st.title("🧮 Regulatory Capital Buffer Simulator")
st.markdown("Simulate capital needs under IFRS 9 staging and macroeconomic stress.")

# Sidebar Inputs
st.sidebar.header("⚙️ Simulation Settings")
current_capital = st.sidebar.number_input("Current Available Capital (£)", value=500_000)
reg_ratio = st.sidebar.number_input("Regulatory Capital Ratio (%)", value=8.0)
scenario = st.sidebar.selectbox("Stress Scenario", ["baseline", "adverse", "optimistic"])

# Check for required columns
required_cols = [f"ecl_{scenario}", "stage"]
if not all(col in df.columns for col in required_cols):
    st.error(f"Missing required columns for scenario '{scenario}'. Please check ECL simulation output.")
    st.stop()

# Fill missing stage data with 1 (conservative)
df["stage"] = df["stage"].fillna(1)

# Calculate required capital buffer by stage
stage_weights = {1: 0.25, 2: 0.5, 3: 1.0}
capital_requirements = []

for stage, weight in stage_weights.items():
    stage_ecl = df[df["stage"] == stage][f"ecl_{scenario}"].sum()
    capital_requirements.append({
        "Stage": f"Stage {stage}",
        "ECL": stage_ecl,
        "Coverage Rate": weight,
        "Required Capital": stage_ecl * weight
    })

results_df = pd.DataFrame(capital_requirements)
total_required = results_df["Required Capital"].sum()
capital_gap = total_required - current_capital

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Total Required Capital Buffer", f"£{total_required:,.0f}")
col2.metric("Capital Gap", f"£{capital_gap:,.0f}", delta_color="inverse")

# Detailed breakdown table
st.markdown("### 📊 Capital Requirement by IFRS Stage")
st.dataframe(results_df.style.format({
    "ECL": "£{:,.0f}",
    "Required Capital": "£{:,.0f}",
    "Coverage Rate": "{:.0%}"
}), use_container_width=True)

# Optional: Pie chart of capital allocation
fig = px.pie(results_df, values="Required Capital", names="Stage", title="Capital Buffer by Stage")
st.plotly_chart(fig, use_container_width=True)

# Raw ECL data toggle
with st.expander("🧾 Show Raw Portfolio Data"):
    st.dataframe(df.head(100))
