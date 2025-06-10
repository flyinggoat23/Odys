import pandas as pd
from pathlib import Path

# Load ECL results
ecl_df = pd.read_csv("results/ecl_results.csv")

# Optional: Load loan metadata (if available) to segment by product/risk category
try:
    loans_df = pd.read_csv("data/loans.csv")
    df = ecl_df.merge(loans_df[["loan_id", "product_type", "risk_category"]], on="loan_id", how="left")
    print("✅ Merged with loan metadata")
except FileNotFoundError:
    print("⚠️ Metadata not found. Proceeding with ECL results only.")
    df = ecl_df.copy()
    df["product_type"] = "Unknown"
    df["risk_category"] = "Unknown"

# Group by stage, product, risk and sum ECL
summary = (
    df.groupby(["stage", "product_type", "risk_category"])
    .agg(
        total_ecl=("ECL", "sum"),
        avg_pd=("pd_score", "mean"),
        count_loans=("loan_id", "count")
    )
    .reset_index()
)

# Save to CSV
Path("results").mkdir(exist_ok=True)
summary.to_csv("results/ecl_provisions.csv", index=False)
print("✅ Saved provisioning summary to results/ecl_provisions.csv")

# Optional: print summary table
print("\n--- Provisioning Summary (Top 10) ---")
print(summary.sort_values("total_ecl", ascending=False).head(10))
