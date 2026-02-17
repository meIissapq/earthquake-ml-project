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

   # Load clean file (assuming 'earthquakes_clean_monthly.csv' exists from Week 1 Step 1)
   df_clean = pd.read_csv(src)

   # Ensure month_date is datetime
   df_clean["month_date"] = pd.to_datetime(df_clean["month_date"])

   # Create grid cell IDs
   df_clean["grid_lat"] = np.floor(df_clean["latitude"] / BIN) * BIN
   df_clean["grid_lon"] = np.floor(df_clean["longitude"] / BIN) * BIN

   # Use hyphens in cell_id as per PkHBwUMGoii6
   df_clean["cell_id"] = (df_clean["grid_lat"].astype(str) + "_" + df_clean["grid_lon"].astype(str))

   # Sort for easier processing
   df_clean = df_clean.sort_values(["cell_id", "month_date"]).reset_index(drop=True)
   df_clean.to_csv(dst, index=False)
   print(f"Saved: {dst}")
   print(f"Rows: {len(df_clean)}, Cols: {len(df_clean.columns)}")

if __name__ == "__main__":
    main()


