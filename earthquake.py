import pandas as pd
import numpy as np
import os
import shutil
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


# Download latest version
path = kagglehub.dataset_download("ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")

print("Path to dataset files:", path)

# Read in dataset
DATA_DIR = Path.home() / ".cache/kagglehub/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/versions/1"
df = pd.read_csv(DATA_DIR / "earthquake_data_tsunami.csv")

# New month_date column and cleaning data
df['month_date'] =  pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")

cols = ["month_date", "latitude", "longitude", "magnitude", "depth", "sig", "mmi", "cdi"]
df_clean = df[cols].copy()

# Export file
out_file = "data/processed/earthquakes_clean_monthly.csv"
df_clean.to_csv(out_file, index=False)

print("Saved:", out_file)

# Load clean file (assuming 'earthquakes_clean_monthly.csv' exists from Week 1 Step 1)
df_clean = pd.read_csv("data/processed/earthquakes_clean_monthly.csv")

# Ensure month_date is datetime
df_clean["month_date"] = pd.to_datetime(df_clean["month_date"])

BIN = 0.5

# Create grid cell IDs
df_clean["grid_lat"] = np.floor(df_clean["latitude"] / BIN) * BIN
df_clean["grid_lon"] = np.floor(df_clean["longitude"] / BIN) * BIN

# Use hyphens in cell_id as per PkHBwUMGoii6
df_clean["cell_id"] = df_clean["grid_lat"].astype(str) + "_" + df_clean["grid_lon"].astype(str)

# Sort for easier processing
df_clean = df_clean.sort_values(["cell_id", "month_date"]).reset_index(drop=True)

feature_rows = []

for cell_id, g in df_clean.groupby("cell_id"):
    g = g.sort_values("month_date").reset_index(drop=True)

    for _, row in g.iterrows():
        month = row["month_date"]

        # Past data
        past = g[g["month_date"] < month]

        last_1m = past[past["month_date"] >= month - pd.DateOffset(months=1)]
        last_3m = past[past["month_date"] >= month - pd.DateOffset(months=3)]
        last_6m = past[past["month_date"] >= month - pd.DateOffset(months=6)]

        # New time windows of 12 months, for new feature c
        last_12m = past[past["month_date"] >= month - pd.DateOffset(months=12)]

        #new feature, months since last quake cb
        if len(past) == 0:
          months_since_last_quake = 999
        else:
          last_quake_date = past["month_date"].max()
          months_since_last_quake = (month.year - last_quake_date.year) * 12 + (month.month - last_quake_date.month)

        #new feature b, trend feature
        trend_3m_minus_6m_avg = len(last_3m) - (len(last_6m) / 2)

        #new feature c, max magnitude in last 12 months
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

            # adding new features from week 4 a,b,c
            "months_since_last_quake": months_since_last_quake,
            "trend_3m_minus_6m_avg": trend_3m_minus_6m_avg,
            "max_mag_last_12m": max_mag_last_12m,

        })

# Output
step4_features = pd.DataFrame(feature_rows)

# Fill NaN values with 0
step4_features = step4_features.fillna(0) #clean features creation stage

# Save with underscores in cell_id
step4_features.to_csv("data/processed/step4_past_features.csv", index=False)
print("step4_past_features.csv created successfully.")

# --- Code from Week 1 Step 2,3 and Week 1 Steps 5,6,7  ---

# Building off step 1 code for df_my 
df_my = df_clean.copy() # Use the df_clean loaded above

# creates two columns of lat and long and assigns to nearest 0.5
df_my["grid_lat"] = np.floor(df_my["latitude"]/BIN) * BIN
df_my["grid_lon"] = np.floor(df_my["longitude"]/ BIN) * BIN

# Creating Cell Id by combining month date
df_my["cell_id"] = df_my["grid_lat"].map(str) + "_" + df_my["grid_lon"].map(str)

# Define create_ml_ready_labels function (from zj6oP8z5FUv7)
def create_ml_ready_labels(df, threshold=6.0):
    # Aggregate to Max Magnitude per Cell per Month
    monthly_data = df.groupby(['cell_id', 'Year', 'Month'])['magnitude'].max().reset_index()

    # Sort to ensure time shifting is correct
    monthly_data = monthly_data.sort_values(['cell_id', 'Year', 'Month'])

    # Next Month Targets
    monthly_data['next_month_max_mag'] = monthly_data.groupby('cell_id')['magnitude'].shift(-1)

    # Next-month probability (y_prob)
    monthly_data['y_prob'] = (monthly_data['next_month_max_mag'] >= threshold).astype(int)

    # Next-month magnitude class (y_class)
    def assign_class(mag):
        if pd.isna(mag) or mag < threshold:
            return -1
        elif 6.0 <= mag <= 6.9:
            return 0
        elif 7.0 <= mag <= 7.9:
            return 1
        else: # 8.0+
            return 2

    monthly_data['y_class'] = monthly_data['next_month_max_mag'].apply(assign_class)

    # Drop the last month for each cell
    final_df_labels = monthly_data.dropna(subset=['next_month_max_mag']).copy()

    return final_df_labels

# Prepare df_my for create_ml_ready_labels and create ml_cell_month_dataset.csv (from -8yJlWP-FeCL)
df_my['month_date'] = pd.to_datetime(df_my['month_date'])
df_my['Year'] = df_my['month_date'].dt.year
df_my['Month'] = df_my['month_date'].dt.month

ml_cell_month_dataset = create_ml_ready_labels(df_my, threshold=7.0)
ml_cell_month_dataset.to_csv("data/processed/ml_cell_month_dataset.csv", index=False)
print("ml_cell_month_dataset.csv created successfully.")

# --- Code from Week 2 Job 1 (cmJkhz4E7kvC, copied to 5df2144f) to create ml_final_dataset.csv ---

# load both files
features = pd.read_csv("data/processed/step4_past_features.csv", parse_dates = ["month_date"])
labels = pd.read_csv("data/processed/ml_cell_month_dataset.csv")


# converting labels month date format to match features
labels["month_date"] = pd.to_datetime(
    dict(year=labels["Year"], month = labels["Month"], day= 1)
)


# keeping columns needed from labels
labels = labels[["cell_id", "month_date", "y_prob", "y_class"]]

# merging features + labels using the matching keys (cell_id, month_date)
# how = inner keeps rows that exist in both tables
final_df = features.merge(labels, on = ["cell_id", "month_date"], how = "inner")

# filling missing features value with 0 (but not changing y prob/y class)
feature_col_final = final_df.columns.difference(["cell_id", "month_date", "y_prob", "y_class"])
final_df[feature_col_final] = final_df[feature_col_final].fillna(0) # clean after merge with labels

# saving final ML dataset
final_df.to_csv("data/processed/ml_final_dataset.csv", index = False)

shutil.copy("data/processed/ml_final_dataset.csv", "data/processed/ml_final_dataset_v2.csv")
print("Copied ml_final_dataset.csv -> ml_final_dataset_v2.csv (optional backup)")



#----Week 2 Job 2----
# 1. Load the ml_final_dataset.csv file
ml_data = pd.read_csv("data/processed/ml_final_dataset.csv", parse_dates=["month_date"])
print(ml_data.columns) #cb

# 2. Define the list of feature columns
feature_columns = [
    "count_last_1m",
    "count_last_3m",
    "count_last_6m",
    "max_mag_last_3m",
    "max_mag_last_6m",
    "avg_mag_last_6m",
    "avg_depth_last_6m",
    "max_depth_last_6m",
    "max_sig_last_3m",
    "avg_mmi_last_3m",
    "avg_cdi_last_3m",
    "months_since_last_quake",
    "trend_3m_minus_6m_avg",
    "max_mag_last_12m",
]

# 3. Create the feature matrix X and target vector Y
X = ml_data[feature_columns]
Y = ml_data["y_prob"]

# 4. Sort the DataFrame by month_date for time-based split
ml_data = ml_data.sort_values("month_date")

# Re-create X and Y after sorting to maintain correct alignment
X = ml_data[feature_columns]
Y = ml_data["y_prob"]

# 5. Determine the split index
split_index = int(len(ml_data) * 0.8)

# 6. Create the training and testing sets
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
Y_train = Y.iloc[:split_index]
Y_test = Y.iloc[split_index:]

print(f"Training data: {len(X_train)} rows")
print(f"Testing data: {len(X_test)} rows")

# 7. Instantiate a RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 8. Train the RandomForestClassifier
model_rf.fit(X_train, Y_train)


#----Week 2 Job 3----

# 1. Load the ml_final_dataset.csv file
df = pd.read_csv("data/processed/ml_final_dataset_v2.csv", parse_dates=["month_date"])

# 2. Filter out invalid classes
df = df[df["y_class"] != -1].copy()

# Sort by month_date for time-based split
df = df.sort_values("month_date")

# Time-based split with safety check
split_ratio = 0.8
split_idx = int(len(df) * split_ratio)

# Ensure split_idx is within bounds
if split_idx >= len(df):
    split_idx = len(df) - 1
if split_idx < 0:
    split_idx = 0

split_date = df.iloc[split_idx]["month_date"]

train_df = df[df["month_date"] < split_date].copy()
test_df = df[df["month_date"] >= split_date].copy()

# Prepare features
exclude_cols = ["cell_id", "month_date", "y_prob", "y_class"]
feature_col = [col for col in df.columns if col not in exclude_cols]


X_train = train_df[feature_col].fillna(0)
y_train = train_df["y_class"]
X_test = test_df[feature_col].fillna(0)
y_test = test_df["y_class"]

# Instantiate and train RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix evaluation
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save model and feature info
joblib.dump(model, "models/magnitude_class_model.pkl")
feature_info = {
    'feature_col': feature_col,
    'class_mapping': {0: "6.0–6.9", 1: "7.0–7.9", 2: "8.0+"}
}
joblib.dump(feature_info, "models/magnitude_model_info.pkl")

magnitude_model = joblib.load("models/magnitude_class_model.pkl")
magnitude_model_info = joblib.load("models/magnitude_model_info.pkl")

feature_columns_mag_model = magnitude_model_info["feature_col"]

#----Week 2 Job 4----
# 1. Find the latest 'month_date' in the ml_data DataFrame
latest_month = ml_data['month_date'].max()
print(f"Latest month identified: {latest_month.strftime('%Y-%m-%d')}")

# 2. Create a new DataFrame by filtering ml_data for the latest month
latest_month_data = ml_data[ml_data['month_date'] == latest_month].copy()
print(f"Data for latest month ({latest_month.strftime('%Y-%m-%d')}) has {len(latest_month_data)} rows.")

# 3. Extract the 'cell_id' column for later use
cell_ids_latest_month = latest_month_data['cell_id']

# 4. Create the feature matrix for prediction, X_predict_latest_month
X_predict_latest_month = latest_month_data[feature_columns]

# 1. Use the model_rf to predict the probabilities of y_prob for the X_predict_latest_month dataset.
# Select the probability of the positive class (index 1).
risk_prob = model_rf.predict_proba(X_predict_latest_month)[:, 1]

# 2. Store these probabilities in a new column named risk_prob within the latest_month_data DataFrame.
latest_month_data['risk_prob'] = risk_prob

# 3. Create a new column named predicted_quake in the latest_month_data DataFrame.
# Assign a value of 1 to predicted_quake if risk_prob is >= 0.3, and 0 otherwise.
latest_month_data['predicted_quake'] = (latest_month_data['risk_prob'] >= 0.3).astype(int)
# 1. Initialize 'predicted_class' column with -1
latest_month_data['predicted_class'] = -1

# 2. Filter for high-risk cells (where predicted_quake is 1)
high_risk_cells = latest_month_data[latest_month_data['predicted_quake'] == 1].copy()

# Check if there are any high-risk cells to predict for
if not high_risk_cells.empty:
    # 3. Extract feature columns required by the magnitude_model
    # The feature_col is available from the loaded magnitude_model_info
    feature_columns_mag_model = magnitude_model_info['feature_col']
    X_high_risk = high_risk_cells[feature_columns_mag_model].fillna(0) # Fill NaN in features with 0

    # 4. Use the loaded magnitude_model to predict the magnitude class
    predicted_classes = magnitude_model.predict(X_high_risk)

    # 5. Update the 'predicted_class' in the original latest_month_data DataFrame
    # Map the predictions back to the original DataFrame using the index
    latest_month_data.loc[high_risk_cells.index, 'predicted_class'] = predicted_classes
    print(f"Magnitude classes predicted for {len(high_risk_cells)} high-risk cells.")
else:
    print("No high-risk cells (predicted_quake = 1) found for magnitude class prediction.")


predictions_df = latest_month_data[['cell_id', 'month_date', 'risk_prob', 'predicted_quake', 'predicted_class']].copy()

# 2. Save the predictions_df DataFrame to a CSV file
predictions_df.to_csv('outputs/predictions_latest_month.csv', index=False)

# 1. Load the 'predictions_latest_month.csv' file into a pandas DataFrame.
predictions_df = pd.read_csv('outputs/predictions_latest_month.csv')

# 2. Sort this DataFrame by the 'risk_prob' column in descending order.
sorted_predictions = predictions_df.sort_values(by='risk_prob', ascending=False)

# 3. Display the 'cell_id' and 'risk_prob' columns for the top 10 rows of the sorted DataFrame.
top_10_risk_cells = sorted_predictions.head(10)[['cell_id', 'risk_prob']]

print("Top 10 cells with the highest risk probability:")
print(top_10_risk_cells)


#----Week 3 Job 2----
models = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
}

thresholds = [0.3, 0.5, 0.7]
rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan

    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "model": name,
            "threshold": t,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": auc,
            "alerts_sent": int(preds.sum())
        })

results_df = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
results_df


#----Week 3 Job 3----
# load dataset
ml_data = pd.read_csv("data/processed/ml_final_dataset.csv")

# keep rows with valid magnitude classes
df_class = ml_data[ml_data["y_class"] != -1].copy()

# convert to 2-class problem if class 2 is rare
# class 0: 6.0-6.9
# class 1: 7.0+
df_class["y_class_2"] = df_class["y_class"].apply(lambda x: 0 if x == 0 else 1)

# train improved magnitude class model
exclude_cols = ["cell_id", "month_date", "y_prob", "y_class", "y_class_2"]
feature_cols = [c for c in df_class.columns if c not in exclude_cols]

X = df_class[feature_cols].fillna(0)
y = df_class["y_class_2"]

# retrain
mag_model_v2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

mag_model_v2.fit(X,y)

# eval with confusion matrix

y_pred = mag_model_v2.predict(X)

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, digits=3))
print(f1_score(y, y_pred, average=None))

# save updated model
joblib.dump(mag_model_v2, "models/magnitude_class_model_v2.pkl")

mag_model_info_v2 = {
    "feature_col": feature_cols,
    "class_mapping": {
        0: "6.0–6.9",
        1: "7.0+"
    },
    "model_type": "2-class RandomForest"
}

joblib.dump(mag_model_info_v2, "models/magnitude_model_info_v2.pkl")


#----Week 3 Job 4----
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

