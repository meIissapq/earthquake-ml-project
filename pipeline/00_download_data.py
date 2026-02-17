import shutil
import kagglehub
from pathlib import Path

def main():
    path = kagglehub.dataset_download(
        "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
    )
    print("Path to dataset files:", path)

    DATA_DIR = Path.home() / ".cache/kagglehub/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/versions/1"

    BASE_DIR = Path(__file__).resolve().parents[1]
    RAW_DIR = BASE_DIR / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    src = DATA_DIR / "earthquake_data_tsunami.csv"
    dst = RAW_DIR / "earthquake_data_tsunami.csv"

    shutil.copy(src, dst)
    print("Saved raw file to:", dst)

if __name__ == "__main__":
    main()