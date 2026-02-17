import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib


def main():
    # --- Week 2 Job 3 ---
    # 1. Load the ml_final_dataset.csv file
    df = pd.read_csv(
        "data/processed/ml_final_dataset.csv",
        parse_dates=["month_date"]
    )

    # 2. Filter out invalid classes
    df = df[df["y_class"] != -1].copy()

    # 3. Sort by month_date for time-based split
    df = df.sort_values("month_date")

    # Time-based split with safety check
    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)

    # Ensure split_idx is within bounds
    if split_idx >= len(df):
        split_idx = len(df) - 1
    if split_idx < 0:
        split_idx = 0

    split_date = df.iloc[split_idx]["month_date"]

    train_df = df[df["month_date"] < split_date].copy()
    test_df = df[df["month_date"] >= split_date].copy()

    print(f"Training data: {len(train_df)} rows")
    print(f"Testing data: {len(test_df)} rows")

    # 4. Prepare features
    exclude_cols = ["cell_id", "month_date", "y_prob", "y_class"]
    feature_col = [col for col in df.columns if col not in exclude_cols]

    X_train = train_df[feature_col].fillna(0)
    y_train = train_df["y_class"]
    X_test = test_df[feature_col].fillna(0)
    y_test = test_df["y_class"]

    # 5. Train RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Confusion matrix evaluation
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 7. Save model and feature info
    joblib.dump(model, "models/magnitude_class_model.pkl")

    feature_info = {
        "feature_col": feature_col,
        "class_mapping": {0: "6.0-6.9", 1: "7.0-7.9", 2: "8.0+"},
    }
    joblib.dump(feature_info, "models/magnitude_model_info.pkl")

    print("Saved: models/magnitude_class_model.pkl")
    print("Saved: models/magnitude_model_info.pkl")


if __name__ == "__main__":
    main()
