import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Earthquake Risk Predictor", layout="wide")

st.title("Earthquake Risk & Depth Visualizer + Predictor")
st.write("Shows grid-cell risk predictions for the latest month from the ML pipeline.")

PRED_PATH = "outputs/predictions_latest_month.csv"

@st.cache_data
def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # If your file doesn't have lat/lon yet, we can derive from cell_id (common format: "lat_lon")
    if "cell_id" in df.columns and ("grid_lat" not in df.columns or "grid_lon" not in df.columns):
        parts = df["cell_id"].astype(str).str.split("_", expand=True)
        if parts.shape[1] >= 2:
            df["grid_lat"] = pd.to_numeric(parts[0], errors="coerce")
            df["grid_lon"] = pd.to_numeric(parts[1], errors="coerce")

    return df

try:
    preds = load_preds(PRED_PATH)
except FileNotFoundError:
    st.error(f"Could not find {PRED_PATH}. Run your pipeline first to generate predictions.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
if "risk_prob" in preds.columns:
    thresh = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.5, 0.01)
else:
    thresh = st.sidebar.slider("Risk threshold (risk_prob column missing)", 0.0, 1.0, 0.5, 0.01)

# Filter high risk
if "risk_prob" in preds.columns:
    high = preds[preds["risk_prob"] >= thresh].copy()
else:
    high = preds.copy()

# Alerts
st.subheader("Alerts")
if len(high) == 0:
    st.info("No cells above the current threshold.")
else:
    # Show a few alerts
    show_n = min(10, len(high))
    for _, row in high.sort_values("risk_prob", ascending=False).head(show_n).iterrows():
        cell = row.get("cell_id", "unknown_cell")
        prob = row.get("risk_prob", None)
        mag_class = row.get("predicted_class", row.get("y_class_pred", ""))
        if prob is not None:
            st.write(f"• High risk ({prob:.2f}) in cell {cell}, predicted class: {mag_class}")
        else:
            st.write(f"• High risk in cell {cell}, predicted class: {mag_class}")

# Map
st.subheader("Map")
needed = {"grid_lat", "grid_lon"}
if needed.issubset(preds.columns) and "risk_prob" in preds.columns:
    fig = px.scatter_geo(
        preds,
        lat="grid_lat",
        lon="grid_lon",
        size="risk_prob",
        hover_name="cell_id" if "cell_id" in preds.columns else None,
        hover_data=["risk_prob"] + (["predicted_class"] if "predicted_class" in preds.columns else []),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Map requires columns grid_lat, grid_lon, and risk_prob. If your predictions file lacks them, we can add them in pipeline/08.")

# Table view
st.subheader("Top Risky Cells")
if "risk_prob" in preds.columns:
    st.dataframe(preds.sort_values("risk_prob", ascending=False).head(10))
else:
    st.dataframe(preds.head(10))
