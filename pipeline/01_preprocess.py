from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    src = RAW_DIR / "earthquake_data_tsunami.csv"
    dst = PROCESSED_DIR / "earthquakes_clean_monthly.csv"

    df = pd.read_csv(src)

    # build month_date from Year + Month
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    df = df.dropna(subset=["Year", "Month", "latitude", "longitude", "magnitude"])

    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df["month_date"] = pd.to_datetime({
        "year": df["Year"],
        "month": df["Month"],
        "day": 1
    }, errors="coerce")

    df = df.dropna(subset=["month_date"])
    df = df.sort_values("month_date").reset_index(drop=True)

    df.to_csv(dst, index=False)

    print(f"Saved: {dst}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

if __name__ == "__main__":
    main()

