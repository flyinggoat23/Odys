import pandas as pd

REQUIRED_LOAN_COLUMNS = [
    "loan_id", "member_id", "start_date", "term_months", "balance", "product_type"
]

REQUIRED_MEMBER_COLUMNS = [
    "member_id", "age", "income", "employment_status", "history_score"
]

def load_csv(path, required_columns):
    try:
        df = pd.read_csv(path)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {str(e)}")

def load_data():
    loans = load_csv("data/loans.csv", REQUIRED_LOAN_COLUMNS)
    members = load_csv("data/members.csv", REQUIRED_MEMBER_COLUMNS)
    return loans, members
