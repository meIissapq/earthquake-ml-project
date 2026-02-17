# Earthquake Risk & Depth Visualizer + Predictor

## What This Project Does

- Downloads and preprocesses historical earthquake data
- Builds grid-based monthly features and labels
- Trains Random Forest models for probability prediction and magnitude classification
- Generates predictions for the latest available month and evaluates performance

## Dataset Used

Historical earthquake dataset from Kaggle:

https://www.kaggle.com/

(Downloaded programmatically in `00_download_data.py`)

## Real-Time Data Source

IRIS SeedLink (real-time seismic data source):

https://ds.iris.edu/ds/nodes/dmc/services/seedlink/

## Models Used

- **Random Forest (Probability Model)** — predicts earthquake occurrence probability (`y_prob`)
- **Random Forest (Magnitude Classification Model)** — predicts earthquake magnitude class (`y_class`)


## How to Run

# 1) Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Outputs

After running the full pipeline, the following files will be generated:

- `outputs/predictions_latest_month.csv`
- `outputs/eval_yprob_report.txt`
- `outputs/eval_yclass_report.txt`
- `outputs/eval_yclass_confusion_matrix.txt`

Note: The prediction file may contain only one row if the latest month in the dataset contains only one grid cell-month entry. This is expected behavior and not an error.

