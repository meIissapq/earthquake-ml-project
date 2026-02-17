import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():

    # --- Week 2 Job 2 ---
    # 1. Load the ml_final_dataset.csv file
    ml_data = pd.read_csv(
        "data/processed/ml_final_dataset.csv",
        parse_dates=["month_date"]
    )

    # 2. Define the list of feature columns
    feature_columns = [
        "count_last_1m",
        "count_last_3m",
        "count_last_6m",
        "max_mag_last_3m",
        "max_mag_last_6m",
        "avg_mag_last_6m",
        "avg_depth_last_6m",
        "max_depth_last_6m",
        "max_sig_last_3m",
        "avg_mmi_last_3m",
        "avg_cdi_last_3m",
        "months_since_last_quake",
        "trend_3m_minus_6m_avg",
        "max_mag_last_12m",
    ]

    # 3. Create the feature matrix X and target vector Y
    X = ml_data[feature_columns]
    Y = ml_data["y_prob"]

    # 4. Sort by month_date for time-based split
    ml_data = ml_data.sort_values("month_date")

    # Re-create X and Y after sorting
    X = ml_data[feature_columns]
    Y = ml_data["y_prob"]

    # 5. Determine the split index
    split_index = int(len(ml_data) * 0.8)

    # 6. Create the training and testing sets
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    Y_train = Y.iloc[:split_index]
    Y_test = Y.iloc[split_index:]

    print(f"Training data: {len(X_train)} rows")
    print(f"Testing data: {len(X_test)} rows")

    # 7. Instantiate a RandomForestClassifier
    model_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # 8. Train the RandomForestClassifier
    model_rf.fit(X_train, Y_train)

    # Save model
    joblib.dump(model_rf, "models/rf_yprob_model.joblib")
    print("Model saved to models/rf_yprob_model.joblib")


if __name__ == "__main__":
    main()
