import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def create_ml_ready_labels(df, threshold=6.0):
    monthly_data = df.groupby(['cell_id', 'Year', 'Month'])['magnitude'].max().reset_index()
    monthly_data = monthly_data.sort_values(['cell_id', 'Year', 'Month'])
    monthly_data['next_month_max_mag'] = monthly_data.groupby('cell_id')['magnitude'].shift(-1)
    monthly_data['y_prob'] = (monthly_data['next_month_max_mag'] >= threshold).astype(int)

    def assign_class(mag):
        if pd.isna(mag) or mag < threshold:
            return -1
        elif 6.0 <= mag <= 6.9:
            return 0
        elif 7.0 <= mag <= 7.9:
            return 1
        else:
            return 2

    monthly_data['y_class'] = monthly_data['next_month_max_mag'].apply(assign_class)
    final_df_labels = monthly_data.dropna(subset=['next_month_max_mag']).copy()
    return final_df_labels

def main():
    src = PROCESSED_DIR / "earthquakes_with_cells.csv"
    dst = PROCESSED_DIR / "ml_cell_month_dataset.csv"

    df_my = pd.read_csv(src, parse_dates=["month_date"])
    df_my["Year"] = df_my["month_date"].dt.year
    df_my["Month"] = df_my["month_date"].dt.month

    ml_cell_month_dataset = create_ml_ready_labels(df_my, threshold=7.0)
    ml_cell_month_dataset.to_csv(dst, index=False)

    print("ml_cell_month_dataset.csv created successfully.")
    print(f"Saved: {dst}")
    print(f"Rows: {len(ml_cell_month_dataset)}, Cols: {len(ml_cell_month_dataset.columns)}")

if __name__ == "__main__":
    main()

