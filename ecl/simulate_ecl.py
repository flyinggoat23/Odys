import pandas as pd
import numpy as np
import joblib

# Load data
loans = pd.read_csv("data/loans.csv")
behavior = pd.read_csv("data/behavior_features.csv")
model = joblib.load("pd/pd_model.joblib")

# Merge data
df = loans.merge(behavior, on="loan_id", how="left")

# Fill missing behavioral data (safe defaults)
df.fillna({
    "missed_payments": 0,
    "avg_payment_ratio": 1.0,
    "payment_volatility": 0.0,
    "last_payment_gap": 0
}, inplace=True)

# Feature list (must match model)
features = [
    "loan_amount", "term_months", "interest_rate",
    "missed_payments", "avg_payment_ratio",
    "payment_volatility", "last_payment_gap"
]

X = df[features]

# Predict PD (probability of default)
df["pd"] = model.predict_proba(X)[:, 1]

# Macro scenarios
scenarios = {
    "baseline": 1.0,
    "adverse": 1.5,
    "optimistic": 0.7
}

# Assume LGD (loss given default) and EAD (exposure at default)
df["lgd"] = 0.45  # Conservative default assumption
df["ead"] = df["loan_amount"]

# Simulate ECL under each scenario
for name, weight in scenarios.items():
    df[f"ecl_{name}"] = df["pd"] * weight * df["lgd"] * df["ead"]

# Weighted average ECL
df["ecl_weighted"] = (
    0.6 * df["ecl_baseline"] +
    0.3 * df["ecl_adverse"] +
    0.1 * df["ecl_optimistic"]
)

# Rename for downstream compatibility
df.rename(columns={
    "pd": "pd_score",
    "ecl_weighted": "ecl"
}, inplace=True)

# Output
df.to_csv("ecl/ecl_results.csv", index=False)
print("✅ Done: Simulated ECL and saved to ecl/ecl_results.csv")
