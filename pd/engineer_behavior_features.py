import pandas as pd

# Load repayments
repayments = pd.read_csv("data/repayments.csv")

# Calculate payment ratio
repayments["payment_ratio"] = repayments["amount_paid"] / repayments["scheduled_payment"]

# Convert dates
repayments["due_date"] = pd.to_datetime(repayments["due_date"])

# Feature engineering
def engineer_behavior_features(df):
    grouped = df.groupby("loan_id")

    features = grouped.agg(
        missed_payments=("payment_ratio", lambda x: (x < 0.1).sum()),
        avg_payment_ratio=("payment_ratio", "mean"),
        payment_volatility=("payment_ratio", "std"),
        last_payment_gap=("due_date", lambda x: (
            (pd.Timestamp.today() - x[x.index[-1]]).days if not x.empty else 0
        ))
    ).reset_index()

    return features

# Create feature set
behavior_df = engineer_behavior_features(repayments)

# Save to CSV
behavior_df.to_csv("data/behavior_features.csv", index=False)
print("✅ Done: Saved behavioral features.")
