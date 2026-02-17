import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ml_path = PROCESSED_DIR / "ml_final_dataset.csv"
    df = pd.read_csv(ml_path, parse_dates=["month_date"]).sort_values("month_date")

    feature_columns = [
        "count_last_1m","count_last_3m","count_last_6m",
        "max_mag_last_3m","max_mag_last_6m","avg_mag_last_6m",
        "avg_depth_last_6m","max_depth_last_6m",
        "max_sig_last_3m","avg_mmi_last_3m","avg_cdi_last_3m",
        "months_since_last_quake","trend_3m_minus_6m_avg","max_mag_last_12m",
    ]

    X = df[feature_columns]
    y = df["y_prob"]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model_rf.fit(X_train, y_train)

    joblib.dump(model_rf, MODELS_DIR / "model_y_prob.pkl")
    joblib.dump(feature_columns, MODELS_DIR / "yprob_feature_columns.pkl")

    y_pred = model_rf.predict(X_test)
    report = classification_report(y_test, y_pred)

    (OUTPUTS_DIR / "yprob_metrics.txt").write_text(report)
    print("Saved model: models/model_y_prob.pkl")
    print("Saved metrics: outputs/yprob_metrics.txt")

if __name__ == "__main__":
    main()
