from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(PROCESSED_DIR / "ml_final_dataset.csv", parse_dates=["month_date"])
    df = df.sort_values("month_date").reset_index(drop=True)

    # ---------- Load models ----------
    yprob_model = joblib.load(MODELS_DIR / "model_y_prob.pkl")
    yprob_feature_cols = joblib.load(MODELS_DIR / "yprob_feature_columns.pkl")

    yclass_model = joblib.load(MODELS_DIR / "magnitude_class_model.pkl")
    yclass_info = joblib.load(MODELS_DIR / "magnitude_model_info.pkl")
    yclass_feature_cols = yclass_info["feature_col"]

    # ---------- Evaluate y_prob ----------
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_prob = test_df[yprob_feature_cols].fillna(0)
    y_prob_true = test_df["y_prob"]

    y_prob_pred = yprob_model.predict(X_prob)
    prob_report = classification_report(y_prob_true, y_prob_pred, zero_division=0)



    (OUTPUTS_DIR / "eval_yprob_report.txt").write_text(prob_report)

    # ---------- Evaluate y_class (only rows where y_class != -1) ----------
    df_class = df[df["y_class"] != -1].copy()
    df_class = df_class.sort_values("month_date").reset_index(drop=True)

    split_idx_c = int(len(df_class) * 0.8)
    test_c = df_class.iloc[split_idx_c:].copy()

    X_class = test_c[yclass_feature_cols].fillna(0)
    y_class_true = test_c["y_class"]

    y_class_pred = yclass_model.predict(X_class)
    class_report = classification_report(y_class_true, y_class_pred)

    cm = confusion_matrix(y_class_true, y_class_pred)

    (OUTPUTS_DIR / "eval_yclass_report.txt").write_text(class_report)
    (OUTPUTS_DIR / "eval_yclass_confusion_matrix.txt").write_text(str(cm))

    print("Saved:")
    print(" - outputs/eval_yprob_report.txt")
    print(" - outputs/eval_yclass_report.txt")
    print(" - outputs/eval_yclass_confusion_matrix.txt")

if __name__ == "__main__":
    main()
