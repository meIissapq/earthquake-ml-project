from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dst = PROCESSED_DIR / "earthquakes_clean_monthly.csv"

    df = pd.read_csv(RAW_DIR / "earthquake_data_tsunami.csv")

    # Build month_date from Year + Month
    df['month_date'] =  pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")

    # Keep only needed columns
    cols = ["month_date", "latitude", "longitude", "magnitude", "depth", "sig", "mmi", "cdi"]
    df_clean = df[cols].copy()

    # Sort
    df_clean = df_clean.sort_values("month_date").reset_index(drop=True)

    df_clean.to_csv(dst, index=False)

    print(f"Saved: {dst}")
    print(f"Rows: {len(df_clean)}, Cols: {len(df_clean.columns)}")

if __name__ == "__main__":
    main()

