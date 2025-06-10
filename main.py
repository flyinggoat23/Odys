from utils.load_data import load_data
from models.pd_model import simulate_default_flag, train_pd_model
from models.ecl import calculate_ecl

if __name__ == "__main__":
    loans, members = load_data()
    print("✅ Loans shape:", loans.shape)
    print("✅ Members shape:", members.shape)
    print("🔍 Loans preview:\n", loans.head())


if __name__ == "__main__":
    loans, members = load_data()

    merged_df = simulate_default_flag(loans, members)
    model, results = train_pd_model(merged_df)

    print("✅ Sample PD scores:")
    print(results.head())

    results.to_csv("reports/pd_scores.csv", index=False)

if __name__ == "__main__":
    loans, members = load_data()
    merged_df = simulate_default_flag(loans, members)
    model, pd_results = train_pd_model(merged_df)

    ecl_table = calculate_ecl(pd_results.merge(loans[["loan_id", "balance"]], on="loan_id"))
    print("✅ ECL table preview:")
    print(ecl_table.head())

    ecl_table.to_csv("reports/ecl_table.csv", index=False)