import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    src = PROCESSED_DIR / "earthquakes_with_cells.csv"
    dst = PROCESSED_DIR / "step4_past_features.csv"

    df_clean = pd.read_csv(src, parse_dates=["month_date"])
    df_clean = df_clean.sort_values(["cell_id", "month_date"]).reset_index(drop=True)

    feature_rows = []

    for cell_id, g in df_clean.groupby("cell_id"):
        g = g.sort_values("month_date").reset_index(drop=True)

        for _, row in g.iterrows():
            month = row["month_date"]

            past = g[g["month_date"] < month]
            last_1m = past[past["month_date"] >= month - pd.DateOffset(months=1)]
            last_3m = past[past["month_date"] >= month - pd.DateOffset(months=3)]
            last_6m = past[past["month_date"] >= month - pd.DateOffset(months=6)]
            last_12m = past[past["month_date"] >= month - pd.DateOffset(months=12)]

            # new feature: months since last quake
            if len(past) == 0:
                months_since_last_quake = 999
            else:
                last_quake_date = past["month_date"].max()
                months_since_last_quake = (month.year - last_quake_date.year) * 12 + (month.month - last_quake_date.month)

            # new feature: trend
            trend_3m_minus_6m_avg = len(last_3m) - (len(last_6m) / 2)

            # new feature: max mag last 12 months
            if len(last_12m) == 0:
                max_mag_last_12m = 0
            else:
                max_mag_last_12m = last_12m["magnitude"].max()

            feature_rows.append({
                "cell_id": cell_id,
                "month_date": month,

                # Count features
                "count_last_1m": len(last_1m),
                "count_last_3m": len(last_3m),
                "count_last_6m": len(last_6m),

                # Magnitude features
                "max_mag_last_3m": last_3m["magnitude"].max(),
                "max_mag_last_6m": last_6m["magnitude"].max(),
                "avg_mag_last_6m": last_6m["magnitude"].mean(),

                # Depth features
                "avg_depth_last_6m": last_6m["depth"].mean(),
                "max_depth_last_6m": last_6m["depth"].max(),

                # Quality / impact features
                "max_sig_last_3m": last_3m["sig"].max(),
                "avg_mmi_last_3m": last_3m["mmi"].mean(),
                "avg_cdi_last_3m": last_3m["cdi"].mean(),

                # Week 4 features
                "months_since_last_quake": months_since_last_quake,
                "trend_3m_minus_6m_avg": trend_3m_minus_6m_avg,
                "max_mag_last_12m": max_mag_last_12m,
            })

    step4_features = pd.DataFrame(feature_rows).fillna(0)
    step4_features.to_csv(dst, index=False)

    print("step4_past_features.csv created successfully.")
    print(f"Saved: {dst}")
    print(f"Rows: {len(step4_features)}, Cols: {len(step4_features.columns)}")

if __name__ == "__main__":
    main()
