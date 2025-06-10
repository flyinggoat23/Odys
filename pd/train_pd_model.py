import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import date, timedelta
import joblib
import os

def generate_synthetic_loan_data(n_loans=1000):
    """Generate realistic synthetic loan portfolio data"""
    np.random.seed(42)
    
    loan_ids = [f"L{str(i).zfill(5)}" for i in range(1, n_loans+1)]
    
    data = {
        "loan_id": loan_ids,
        "loan_amount": np.round(np.random.lognormal(10, 0.5, n_loans), 2),  # Added loan_amount
        "term_months": np.random.choice([12, 24, 36, 48, 60], n_loans, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        "risk_category": np.random.choice(["A", "B", "C"], n_loans, p=[0.6, 0.3, 0.1]),
        "product_type": np.random.choice(["Personal", "Auto", "Mortgage"], n_loans, p=[0.5, 0.3, 0.2]),
        "interest_rate": np.round(np.random.uniform(0.03, 0.15, n_loans), 4),
        "origination_date": [
            date(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095)) 
            for _ in range(n_loans)
        ],
        "stage": np.random.choice([1, 2, 3], n_loans, p=[0.8, 0.15, 0.05]),
        "missed_payments": np.random.poisson(0.5, n_loans),
        "avg_payment_ratio": np.random.normal(1.0, 0.3, n_loans).clip(0.1, 2.0),
        "payment_volatility": np.random.exponential(0.2, n_loans),
        "last_payment_gap": np.random.randint(0, 90, n_loans),
        "credit_score": np.random.randint(300, 850, n_loans),
        "dti_ratio": np.round(np.random.normal(0.35, 0.15, n_loans).clip(0.05, 0.95), 4)
    }
    
    df = pd.DataFrame(data)
    df["origination_date"] = pd.to_datetime(df["origination_date"])
    df["age_months"] = (pd.Timestamp.today() - df["origination_date"]).dt.days // 30
    
    # Add realistic relationships
    df.loc[df["risk_category"] == "C", "interest_rate"] *= 1.5
    df.loc[df["stage"] == 3, "avg_payment_ratio"] *= 0.7
    
    # Add at the end before return:
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

# MAIN CODE FLOW START ---------------------------------------------------------

# Try to load pre-merged data first
try:
    df = pd.read_csv("data/processed/merged_data.csv")
    print(f"✅ Loaded merged data with shape: {df.shape}")
except FileNotFoundError:
    print("⚠️ Merged data file not found. Building from components...")
    
    # Try to load loans data
    try:
        loans = pd.read_csv("data/loans.csv")
        print(f"✅ Loaded loans data: {loans.shape}")
    except FileNotFoundError:
        print("⚠️ Loans file not found. Generating synthetic loans...")
        loans = generate_synthetic_loan_data(500)
        print(f"🔄 Created synthetic loans: {loans.shape}")
    
    # Try to load behavior data
    try:
        behavior = pd.read_csv("data/behavior_features.csv")
        print(f"✅ Loaded behavior data: {behavior.shape}")
    except FileNotFoundError:
        print("⚠️ Behavior file not found. Generating placeholder...")
        # Create minimal behavior data
        behavior = pd.DataFrame({
            "loan_id": loans["loan_id"],
            "missed_payments": np.zeros(len(loans)),
            "avg_payment_ratio": np.ones(len(loans)),
            "payment_volatility": np.zeros(len(loans)),
            "last_payment_gap": np.zeros(len(loans))
        })
    
    # Merge datasets
    df = loans.merge(behavior, on="loan_id", how="left")
    print(f"✅ Merged data: {df.shape}")

# Data conversion and cleaning
num_cols = ["missed_payments", "avg_payment_ratio", "payment_volatility", "last_payment_gap"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"⚠️ Column {col} missing. Creating with default values.")
        df[col] = 0  # Default value

# Fill missing values
fill_values = {
    "missed_payments": 0,
    "avg_payment_ratio": 1.0,
    "payment_volatility": 0.0,
    "last_payment_gap": 0
}
df.fillna(fill_values, inplace=True)

# Create target variable
if "stage" in df.columns:
    df["target"] = df["stage"].apply(lambda x: 1 if x in [2, 3] else 0)
    print(f"✅ Created target. Default rate: {df['target'].mean():.2%}")
else:
    print("⚠️ 'stage' column missing. Creating random target")
    df["target"] = np.random.randint(0, 2, len(df))

# Data validation
if len(df) < 5:
    print(f"❌ Critical: Only {len(df)} samples. Augmenting with synthetic data")
    extra = generate_synthetic_loan_data(100)
    df = pd.concat([df, extra], ignore_index=True)
    print(f"🆕 New data size: {len(df)}")
    
    # After augmentation
    if "stage" in df.columns:
        # Create/recreate target for entire DataFrame
        df["target"] = df["stage"].apply(lambda x: 1 if x in [2, 3] else 0)
        print(f"✅ Recreated target after augmentation. Default rate: {df['target'].mean():.2%}")
    else:
        print("⚠️ 'stage' column missing after augmentation. Creating random target")
        df["target"] = np.random.randint(0, 2, len(df))

# Verify no NaNs in target
if df["target"].isna().any():
    print("⚠️ NaN values found in target. Filling with 0")
    df["target"].fillna(0, inplace=True)
    
# Feature selection
features = ["loan_amount", "term_months", "interest_rate", 
            "missed_payments", "avg_payment_ratio", 
            "payment_volatility", "last_payment_gap"]

# Ensure features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"⚠️ Missing features: {missing_features}. Creating with default values.")
    for f in missing_features:
        df[f] = 0  # Default value

X = df[features]
y = df["target"]

# Validate and clean features
print("\n🔍 Feature Validation:")
nan_counts = X.isna().sum()
print("NaN counts per feature:")
print(nan_counts[nan_counts > 0])

# Fill NaN values
if X.isna().any().any():
    print("⚠️ NaN values found in features. Filling with median values")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Verify fix
    print(f"✅ NaN after imputation: {X.isna().sum().sum()}")
else:
    print("✅ No NaN values found in features")

# Convert all to numeric (in case of mixed types)
X = X.apply(pd.to_numeric, errors='coerce')
if X.isna().any().any():
    print("⚠️ Coercion created new NaN values. Filling with 0")
    X.fillna(0, inplace=True)

# Handle small datasets
MIN_SAMPLES_FOR_SPLIT = 5

if len(X) < MIN_SAMPLES_FOR_SPLIT:
    print(f"⚠️ Insufficient samples ({len(X)}). Using all data for training.")
    X_train, y_train = X, y
    X_test, y_test = None, None
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")

# Train model
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=2000))
])
model.fit(X_train, y_train)

# Save model
os.makedirs("pd", exist_ok=True)
joblib.dump(model, "pd/pd_model.joblib")
print("✅ Model saved to pd/pd_model.joblib")

# Evaluate if test set exists
if X_test is not None and y_test is not None:
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    if hasattr(model, 'coef_'):
        print("\nFeature Importance:")
        for name, importance in zip(features, model.coef_[0]):
            print(f"{name}: {importance:.4f}")
else:
    print("⚠️ No test set available. Model trained on full dataset.")