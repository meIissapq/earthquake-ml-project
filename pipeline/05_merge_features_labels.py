import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    features_path = PROCESSED_DIR / "step4_past_features.csv"
    labels_path   = PROCESSED_DIR / "ml_cell_month_dataset.csv"
    out_path      = PROCESSED_DIR / "ml_final_dataset.csv"
    features = pd.read_csv(features_path, parse_dates=["month_date"])
    labels = pd.read_csv(labels_path, parse_dates=["month_date"])

    # keep only needed columns
    labels = labels[["cell_id", "month_date", "y_prob", "y_class"]]

    final_df = features.merge(labels, on=["cell_id", "month_date"], how="inner")

    # fill feature NaNs with 0 (do NOT touch y columns)
    feature_cols = final_df.columns.difference(["cell_id", "month_date", "y_prob", "y_class"])
    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    final_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(f"Rows: {len(final_df)}, Cols: {len(final_df.columns)}")

if __name__ == "__main__":
    main()
