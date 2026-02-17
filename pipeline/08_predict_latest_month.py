import pandas as pd
import joblib
from pathlib import Path

# Paths (pipeline/ is one level below repo root)
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) Load processed dataset
    ml_data = pd.read_csv(
        PROCESSED_DIR / "ml_final_dataset.csv",
        parse_dates=["month_date"],
    )

    # 2) Load models + feature lists
    model_rf = joblib.load(MODELS_DIR / "rf_yprob_model.joblib")
    feature_columns = joblib.load(MODELS_DIR / "yprob_feature_columns.pkl")


    magnitude_model = joblib.load(MODELS_DIR / "magnitude_class_model.pkl")
    magnitude_model_info = joblib.load(MODELS_DIR / "magnitude_model_info.pkl")

    # 3) Find the latest month in the dataset and filter to it
    latest_month = ml_data["month_date"].max()
    print(f"Latest month identified: {latest_month.strftime('%Y-%m-%d')}")

    latest_month_data = ml_data[ml_data["month_date"] == latest_month].copy()
    print(
        f"Data for latest month ({latest_month.strftime('%Y-%m-%d')}) has {len(latest_month_data)} rows."
    )

    # 4) Predict quake risk probability (y_prob)
    X_predict_latest_month = latest_month_data[feature_columns]
    probs = model_rf.predict_proba(X_predict_latest_month)

    if probs.shape[1] == 1:
    # Model trained on only one class
        if model_rf.classes_[0] == 1:
            risk_prob = probs[:, 0]
        else:
            risk_prob = [0.0] * len(X_predict_latest_month)
    else:
        risk_prob = probs[:, 1]

    latest_month_data["risk_prob"] = risk_prob

    # 5) Convert probability to predicted quake flag (threshold 0.3)
    latest_month_data["predicted_quake"] = (latest_month_data["risk_prob"] >= 0.3).astype(
        int
    )

    # 6) Default magnitude class to -1, then fill only for high-risk rows
    latest_month_data["predicted_class"] = -1

    high_risk_cells = latest_month_data[latest_month_data["predicted_quake"] == 1].copy()

    if not high_risk_cells.empty:
        # Features needed by magnitude model
        feature_columns_mag_model = magnitude_model_info["feature_col"]

        X_high_risk = high_risk_cells[feature_columns_mag_model].fillna(0)
        predicted_classes = magnitude_model.predict(X_high_risk)

        # Map predictions back into latest_month_data
        latest_month_data.loc[high_risk_cells.index, "predicted_class"] = predicted_classes

        print(
            f"Magnitude classes predicted for {len(high_risk_cells)} high-risk cells."
        )
    else:
        print("No high-risk cells (predicted_quake = 1) found for magnitude class prediction.")

    # 7) Build final output dataframe + split cell_id into lat/lon
    predictions_df = latest_month_data[["cell_id", "risk_prob", "predicted_class"]].copy()

    predictions_df[["grid_lat", "grid_lon"]] = predictions_df["cell_id"].str.split(
        "_", expand=True
    )
    predictions_df["grid_lat"] = predictions_df["grid_lat"].astype(float)
    predictions_df["grid_lon"] = predictions_df["grid_lon"].astype(float)

    # Order columns to match expected output
    predictions_df = predictions_df[["grid_lat", "grid_lon", "risk_prob", "predicted_class"]]

    # 8) Save
    out_path = OUTPUTS_DIR / "predictions_latest_month.csv"
    predictions_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(predictions_df)}")


if __name__ == "__main__":
    main()
