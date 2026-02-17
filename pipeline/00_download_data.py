from pathlib import Path
import shutil
import kagglehub

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dataset_path = kagglehub.dataset_download(
        "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
    )

    print(f"Dataset downloaded to: {dataset_path}")

    dataset_path = Path(dataset_path)
    csv_files = list(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")

    src = csv_files[0]
    dst = RAW_DIR / "earthquake_data_tsunami.csv"

    shutil.copy(src, dst)

    print(f"Saved raw file to: {dst}")

if __name__ == "__main__":
    main()




