import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ml_data = pd.read_csv(PROCESSED_DIR / "ml_final_dataset.csv", parse_dates=["month_date"])

    model_rf = joblib.load(MODELS_DIR / "model_y_prob.pkl")
    feature_columns = joblib.load(MODELS_DIR / "yprob_feature_columns.pkl")

    magnitude_model = joblib.load(MODELS_DIR / "magnitude_class_model.pkl")
    magnitude_model_info = joblib.load(MODELS_DIR / "magnitude_model_info.pkl")

    latest_month = ml_data["month_date"].max()
    latest_month_data = ml_data[ml_data["month_date"] == latest_month].copy()

    X_predict_latest_month = latest_month_data[feature_columns]
    risk_prob = model_rf.predict_proba(X_predict_latest_month)[:, 1]
    latest_month_data["risk_prob"] = risk_prob

    latest_month_data["predicted_quake"] = (latest_month_data["risk_prob"] >= 0.3).astype(int)
    latest_month_data["predicted_class"] = -1

    high_risk_cells = latest_month_data[latest_month_data["predicted_quake"] == 1].copy()
    if not high_risk_cells.empty:
        feature_columns_mag_model = magnitude_model_info["feature_col"]
        X_high_risk = high_risk_cells[feature_columns_mag_model].fillna(0)
        predicted_classes = magnitude_model.predict(X_high_risk)
        latest_month_data.loc[high_risk_cells.index, "predicted_class"] = predicted_classes

    predictions_df = latest_month_data[["cell_id", "risk_prob", "predicted_class"]].copy()
    predictions_df[["grid_lat", "grid_lon"]] = predictions_df["cell_id"].str.split("_", expand=True)
    predictions_df["grid_lat"] = predictions_df["grid_lat"].astype(float)
    predictions_df["grid_lon"] = predictions_df["grid_lon"].astype(float)

    predictions_df = predictions_df[["grid_lat", "grid_lon", "risk_prob", "predicted_class"]]
    out_path = OUTPUTS_DIR / "predictions_latest_month.csv"
    predictions_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(predictions_df)}")

if __name__ == "__main__":
    main()

