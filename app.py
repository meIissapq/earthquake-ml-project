import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Seismic Insight Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    
    .alert-card {
        background-color: #161b22;
        border-left: 5px solid #ff4b4b;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 1rem;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
    }
    
    .alert-high { border-left-color: #ff4b4b; }
    .alert-medium { border-left-color: #ffa500; }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPERS ---
def load_data():
    file_path = "outputs/predictions_latest_month.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Split cell_id into lat/lon
    # Expected format: "10.0_72.5"
    df[['lat', 'lon']] = df['cell_id'].str.split('_', expand=True).astype(float)
    
    # Map magnitude class labels
    mag_labels = {
        0: "6.0–6.9",
        1: "7.0–7.9",
        2: "8.0+",
        -1: "N/A"
    }
    df['mag_label'] = df['predicted_class'].map(mag_labels)
    
    return df

# --- MAIN APP ---
st.title("🌍 Seismic Insight Dashboard")
st.markdown("### Predictive Analysis & Real-time Risk Monitoring")
st.markdown("---")

# Load Data
df = load_data()

if df.empty:
    st.warning("No prediction data available to display.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Dashboard Controls")
risk_threshold = st.sidebar.slider(
    "Risk Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.01,
    help="Filter cells and alerts by minimum risk probability."
)

selected_month = st.sidebar.selectbox(
    "Analysis Month",
    options=df['month_date'].unique(),
    index=0
)

# Filter data
filtered_df = df[df['month_date'] == selected_month].copy()
high_risk_df = filtered_df[filtered_df['risk_prob'] >= risk_threshold].sort_values('risk_prob', ascending=False)

# --- TOP STATS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Cells Analyzed", len(filtered_df))
with col2:
    st.metric("High-Risk Zones identified", len(high_risk_df))
with col3:
    max_risk = filtered_df['risk_prob'].max()
    st.metric("Peak Risk Probability", f"{max_risk:.2f}")

# --- MAP DISPLAY ---
st.subheader("Interactive Risk Map")
# Define point color based on risk (Red to Green)
# risk_prob 0 -> Green [0, 255, 0], risk_prob 1 -> Red [255, 0, 0]
filtered_df['r'] = filtered_df['risk_prob'] * 255
filtered_df['g'] = (1 - filtered_df['risk_prob']) * 255
filtered_df['b'] = 0
filtered_df['size'] = filtered_df['risk_prob'] * 50000 + 10000 # radius in meters

layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_df,
    get_position=['lon', 'lat'],
    get_color=['r', 'g', 'b', 180],
    get_radius='risk_prob * 10', # Base radius
    radius_min_pixels=5,         # Ensure small points stay visible
    radius_max_pixels=30,        # Cap large points
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=filtered_df['lat'].mean() if not filtered_df.empty else 0,
    longitude=filtered_df['lon'].mean() if not filtered_df.empty else 0,
    zoom=1.5,
    pitch=0,
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Cell: {cell_id}\nRisk: {risk_prob}\nPredicted Mag: {mag_label}"},
    map_style=None # Fallback to default deck.gl dark style (no Mapbox token needed)
)

st.pydeck_chart(r)

# --- ALERTS & TABLE ---
st.markdown("---")
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("⚠️ Risk Alerts")
    if high_risk_df.empty:
        st.info(f"No cells exceed the threshold of {risk_threshold}")
    else:
        for _, row in high_risk_df.iterrows():
            severity = "high" if row['risk_prob'] >= 0.7 else "medium"
            st.markdown(f"""
            <div class="alert-card alert-{severity}">
                <strong>High risk ({row['risk_prob']:.2f})</strong> in cell {row['cell_id']}<br>
                Predicted Magnitude: {row['mag_label']}
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.subheader("📊 Top 10 Risky Cells")
    top_10 = filtered_df.sort_values('risk_prob', ascending=False).head(10)
    st.table(top_10[['cell_id', 'risk_prob', 'mag_label']].rename(columns={
        'cell_id': 'Cell ID',
        'risk_prob': 'Risk Prob',
        'mag_label': 'Predicted Mag'
    }))

st.markdown("---")
st.caption("Seismic Prediction Engine")
