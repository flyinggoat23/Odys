import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def assign_stage(pd_score):
    if pd_score < 0.05:
        return 1
    elif pd_score < 0.3:
        return 2
    else:
        return 3

def calculate_ecl(df, lgd_rate=0.45, discount_rate=0.05):
    df = df.copy()
    df["LGD"] = lgd_rate
    df["EAD"] = df["balance"]
    df["stage"] = df["pd_score"].apply(assign_stage)
    
    df["discount_factor"] = np.where(
        df["stage"] == 1,
        1 / (1 + discount_rate),
        1 / ((1 + discount_rate) ** 2)
    )
    
    df["ECL"] = df["pd_score"] * df["LGD"] * df["EAD"] * df["discount_factor"]
    return df[["loan_id", "stage", "pd_score", "LGD", "EAD", "discount_factor", "ECL"]]

# Load model
model = joblib.load("pd/pd_model.joblib")

# Load and prepare data
df = pd.read_csv("data/processed/merged_data.csv")

# Clean data: Convert to numeric and handle NaNs
features = ["loan_amount", "term_months", "interest_rate", 
            "missed_payments", "avg_payment_ratio", 
            "payment_volatility", "last_payment_gap"]

for col in features + ["balance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Ensure all features exist
for feature in features:
    if feature not in df.columns:
        df[feature] = 0

# Predict PD scores
X = df[features]
df["pd_score"] = model.predict_proba(X)[:, 1]

# Calculate ECL
ecl_results = calculate_ecl(df)

# Save results
ecl_results.to_csv("results/ecl_results.csv", index=False)
print("✅ ECL calculation complete!")