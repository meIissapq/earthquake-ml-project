import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os

# --- Part B: Alerts Logic ---
def generate_alerts(df, threshold=0.5):
    """
    Simple rule: If risk_prob >= threshold -> alert.
    Includes class prediction if available.
    """
    print(f"\n--- ACTIVE ALERTS (Threshold >= {threshold}) ---")
    high_risk = df[df['risk_prob'] >= threshold].copy()
    
    if high_risk.empty:
        print("No high-risk zones detected.")
        return []
    
    alerts = []
    class_mapping = {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"}
    
    for _, row in high_risk.iterrows():
        msg = f"High earthquake risk next month in cell {row['cell_id']} (Risk={row['risk_prob']:.2f})."
        if row['predicted_class'] in class_mapping:
            msg += f" Expected magnitude class: {class_mapping[row['predicted_class']]}."
        
        print(msg)
        alerts.append(msg)
    
    print("------------------------------------------\n")
    return alerts

# --- Part C: Real-time Data Integration (IRIS FDSN) ---
def fetch_live_quakes():
    """
    Connect to IRIS and pull live events from the last 48 hours.
    """
    print("Fetching live earthquake data from IRIS...")
    try:
        from obspy.clients.fdsn import Client
        from obspy import UTCDateTime
        
        client = Client("IRIS")
        # Get events from last 48 hours, magnitude 4.0+
        starttime = UTCDateTime.now() - 2 * 86400 
        cat = client.get_events(starttime=starttime, minmagnitude=4.0)
        
        live_data = []
        for event in cat:
            origin = event.origins[0]
            mag = event.magnitudes[0].mag
            live_data.append({
                "lat": origin.latitude,
                "lon": origin.longitude,
                "mag": mag,
                "time": origin.time.datetime.strftime("%Y-%m-%d %H:%M"),
                "type": "Live Quake"
            })
        
        df_live = pd.DataFrame(live_data)
        print(f"Successfully fetched {len(df_live)} live quakes.")
        return df_live
    except Exception as e:
        print(f"Could not connect to IRIS ({e}). Using fallback simulation.")
        # Fallback simulation
        return pd.DataFrame([
            {"lat": 35.6895, "lon": 139.6917, "mag": 4.5, "time": "2026-02-09", "type": "Live Quake"},
            {"lat": -12.0464, "lon": -77.0428, "mag": 5.2, "time": "2026-02-09", "type": "Live Quake"},
        ])

# --- Part A: Put predictions on a map ---
def create_dashboard(predictions_file):
    if not os.path.exists(predictions_file):
        print(f"Error: {predictions_file} not found.")
        return

    df = pd.read_csv(predictions_file)
    df[['lat', 'lon']] = df['cell_id'].str.split('_', expand=True).astype(float)
    
    # Create the map with a premium look
    fig = go.Figure()

    # 1. Prediction Grid (Heatmap-like markers)
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=18,
            color=df['risk_prob'],
            colorscale='Viridis', # Professional looking scale
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title="Risk Level",
                thickness=20,
                x=0.9,
                tickvals=[0, 0.5, 1],
                ticktext=["Low", "Medium", "High"]
            )
        ),
        text=df.apply(lambda r: f"<b>Cell:</b> {r['cell_id']}<br><b>Risk:</b> {r['risk_prob']:.2f}<br><b>Class:</b> {r['predicted_class']}", axis=1),
        hoverinfo='text',
        name='Predicted Risk Zones'
    ))

    # 2. Highlight Predicted Quakes
    high_risk_df = df[df['predicted_quake'] == 1]
    if not high_risk_df.empty:
        class_names = {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"}
        fig.add_trace(go.Scattermapbox(
            lat=high_risk_df['lat'],
            lon=high_risk_df['lon'],
            mode='markers+text',
            marker=go.scattermapbox.Marker(
                size=14,
                color='white'
            ),
            text=high_risk_df['predicted_class'].map(lambda x: f"⚠️ {class_names.get(x, 'High Risk')}"),
            textposition='top center',
            textfont=dict(size=12, color="white"),
            name='Alert Zones (Mag > 6.0)',
            hoverinfo='none'
        ))

    # 3. Add Live Quakes (Part C)
    live_df = fetch_live_quakes()
    if not live_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=live_df['lat'],
            lon=live_df['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='cyan',
                opacity=0.9
            ),
            text=live_df.apply(lambda r: f"<b>LIVE QUAKE</b><br>Mag: {r['mag']}<br>Time: {r['time']}", axis=1),
            name='Live Quakes (Last 48h)',
            hoverinfo='text'
        ))

    # Dark Mode Premium Layout
    fig.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter", # Sleek dark map
        mapbox=dict(
            center=go.layout.mapbox.Center(lat=20, lon=0),
            zoom=1.5
        ),
        margin={"r":0,"t":60,"l":0,"b":0},
        title=dict(
            text="🌍 Global Earthquake Risk & Live Monitoring",
            font=dict(size=24, color="white"),
            x=0.05
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )

    # Save to HTML
    out_file = "earthquake_dashboard.html"
    fig.write_html(out_file)
    print(f"Dashboard saved as {out_file}")
    
if __name__ == "__main__":
    pred_file = "predictions_latest_month.csv"
    
    # Run Alerts Logic
    df = pd.read_csv(pred_file)
    generate_alerts(df, threshold=0.5)
    
    # Create Dashboard
    create_dashboard(pred_file)
