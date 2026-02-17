import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(PROCESSED_DIR / "ml_final_dataset.csv", parse_dates=["month_date"])
    df = df[df["y_class"] != -1].copy()
    df = df.sort_values("month_date")

    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)
    split_idx = min(max(split_idx, 0), len(df) - 1)

    split_date = df.iloc[split_idx]["month_date"]
    train_df = df[df["month_date"] < split_date].copy()
    test_df = df[df["month_date"] >= split_date].copy()

    exclude_cols = ["cell_id", "month_date", "y_prob", "y_class"]
    feature_col = [c for c in df.columns if c not in exclude_cols]

    X_train = train_df[feature_col].fillna(0)
    y_train = train_df["y_class"]
    X_test = test_df[feature_col].fillna(0)
    y_test = test_df["y_class"]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    joblib.dump(model, MODELS_DIR / "magnitude_class_model.pkl")
    feature_info = {
        "feature_col": feature_col,
        "class_mapping": {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"},
    }
    joblib.dump(feature_info, MODELS_DIR / "magnitude_model_info.pkl")

    print("Saved: models/magnitude_class_model.pkl")
    print("Saved: models/magnitude_model_info.pkl")

if __name__ == "__main__":
    main()
