import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    src = PROCESSED_DIR / "earthquakes_with_cells.csv"
    dst = PROCESSED_DIR / "ml_cell_month_dataset.csv"

    df = pd.read_csv(src, parse_dates=["month_date"])

    # Max magnitude per cell per month
    monthly = (
        df.groupby(["cell_id", "month_date"])["magnitude"]
        .max()
        .reset_index()
        .sort_values(["cell_id", "month_date"])
    )

    # Next month target
    monthly["next_month_max_mag"] = monthly.groupby("cell_id")["magnitude"].shift(-1)

    # y_prob (binary)
    threshold = 6.0
    monthly["y_prob"] = (monthly["next_month_max_mag"] >= threshold).astype(int)

    # y_class (multi-class)
    def assign_class(mag):
        if pd.isna(mag) or mag < threshold:
            return -1
        elif 6.0 <= mag <= 6.9:
            return 0
        elif 7.0 <= mag <= 7.9:
            return 1
        else:
            return 2

    monthly["y_class"] = monthly["next_month_max_mag"].apply(assign_class)

    # Drop rows where next month doesn't exist
    final_labels = monthly.dropna(subset=["next_month_max_mag"]).copy()

    final_labels.to_csv(dst, index=False)
    print("Saved:", dst)
    print(f"Rows: {len(final_labels)}, Cols: {len(final_labels.columns)}")

if __name__ == "__main__":
    main()
