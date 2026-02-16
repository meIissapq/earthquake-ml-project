# Earthquake Magnitude Classification (ML Project)

## Overview
This project builds a machine learning model to classify earthquake magnitudes using historical earthquake data.

The goal is to:
- Engineer useful temporal and spatial features
- Train classification models to predict magnitude class
- Evaluate model performance
- Generate predictions for the most recent month

---

## Project Structure

earthquake-ml-project/
│
├── earthquake.py                 # Main pipeline script
├── earthquake_backup.py          # Backup version
│
├── earthquakes_clean_monthly.csv # Cleaned monthly earthquake data
├── ml_cell_month_dataset.csv     # Feature engineered dataset
├── ml_final_dataset.csv          # Final training dataset
├── ml_final_dataset_v2.csv       # Updated version of dataset
│
├── magnitude_class_model.pkl     # Trained classification model
├── magnitude_class_model_v2.pkl  # Updated model
├── magnitude_model_info.pkl      # Model metadata
├── magnitude_model_info_v2.pkl   # Updated metadata
│
├── predictions_latest_month.csv  # Model predictions
│
└── .gitignore
