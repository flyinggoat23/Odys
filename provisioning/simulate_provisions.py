import pandas as pd
import numpy as np
from pathlib import Path

# Load ECL results
ecl_path = Path("results/ecl_results.csv")
if not ecl_path.exists():
    raise FileNotFoundError("Run simulate_ecl.py first to generate ECL results.")

df = pd.read_csv(ecl_path)

# ---- ADDED STAGE VALIDATION ----
def clean_stage_column(series):
    """Ensure stage contains only valid integers 1-3"""
    # Replace non-finite values with default stage 1
    cleaned = series.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Convert to numeric and clip to valid range
    cleaned = pd.to_numeric(cleaned, errors='coerce').clip(1, 3)
    
    # Fill any remaining issues with default
    return cleaned.fillna(1).astype(int)

df["stage"] = clean_stage_column(df["stage"])

# ---- END ADDED SECTION ----

# Aggregate provisioning by stage
summary = df.groupby("stage").agg(
    total_ecl=("ecl", "sum"),
    avg_pd=("pd_score", "mean"),
    exposure=("ead", "sum"),
    loans=("loan_id", "count")
).reset_index()

# Add total summary row
total = pd.DataFrame({
    "stage": ["Total"],
    "total_ecl": [df["ecl"].sum()],
    "avg_pd": [df["pd_score"].mean()],
    "exposure": [df["ead"].sum()],
    "loans": [df["loan_id"].nunique()]
})

summary = pd.concat([summary, total], ignore_index=True)

# Save
Path("results").mkdir(exist_ok=True)
summary.to_csv("results/provision_summary.csv", index=False)

print("✅ Provisioning summary saved to results/provision_summary.csv")
print(summary)