import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BIN = 0.5

def main():
    src = PROCESSED_DIR / "earthquakes_clean_monthly.csv"
    dst = PROCESSED_DIR / "earthquakes_with_cells.csv"

    df = pd.read_csv(src, parse_dates=["month_date"])

    df["grid_lat"] = np.floor(df["latitude"] / BIN) * BIN
    df["grid_lon"] = np.floor(df["longitude"] / BIN) * BIN
    df["cell_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    df = df.sort_values(["cell_id", "month_date"]).reset_index(drop=True)

    df.to_csv(dst, index=False)
    print(f"Saved: {dst}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

if __name__ == "__main__":
    main()
