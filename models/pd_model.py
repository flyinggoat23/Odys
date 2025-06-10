import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def simulate_default_flag(loans_df, members_df):
    df = loans_df.merge(members_df, on="member_id")
    # Simulate default: simple rule-based logic
    df["default"] = ((df["balance"] > 3000) & (df["income"] < 20000)).astype(int)
    return df

def train_pd_model(merged_df):
    features = ["age", "income", "history_score", "term_months", "balance"]
    X = merged_df[features]
    y = merged_df["default"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    merged_df["pd_score"] = pipeline.predict_proba(X)[:, 1]  # PD = probability of class 1 (default)

    return pipeline, merged_df[["loan_id", "pd_score", "default"] + features]
